import time
import argparse
import pickle
from model import *
from utils import *
import os
import torch
import datetime
from tqdm import tqdm
from dataset_utils import infer_num_nodes_from_sequences, resolve_dataset_dir
from length_bucket_metrics import format_length_bucket_report, summarize_length_buckets


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='Tmall/retailrocket/lastfm/diginetica/yoochoose1_64/yoochoose1_4')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--interests', type=int, default=3, help='The number of interests')
parser.add_argument('--beta', type=float, default=0.01, help='Beta for the interests regularization')
parser.add_argument('--length', type=float, default=8, help='eta in the paper')
parser.add_argument('--cl_rate', type=float, default=0.01, help='Weight for contrastive loss')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
parser.add_argument('--split_threshold', type=int, default=999, help='Only sessions longer than this trigger splitting')
parser.add_argument('--split_lambda', type=float, default=0.0, help='Fusion weight for split-session scores')

opt = parser.parse_args()


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, return_hidden=False):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden = model(items, adj, mask, inputs)

    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    if return_hidden:
        return targets, model.compute_scores(seq_hidden, mask), seq_hidden, mask
    else:
        return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data, epoch_idx):
    print(f'Start training epoch {epoch_idx}: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=8, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch_idx}', unit='batch', mininterval=0.5)

    for data in pbar:
        model.optimizer.zero_grad()

        targets, (scores, loss1, _), hidden1, mask = forward(model, data, return_hidden=True)
        _, _, hidden2, _ = forward(model, data, return_hidden=True)
        targets = trans_to_cuda(targets).long()
        loss_cl = model.ssl_loss(hidden1, hidden2, mask)
        loss = model.loss_function(scores, targets - 1) + loss1 + opt.cl_rate * loss_cl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        model.optimizer.step()

        total_loss += loss
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    print('\tTotal Loss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()

    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    cov = set()
    lengths = []
    test_pbar = tqdm(test_loader, desc='Predicting', unit='batch')

    for data in test_pbar:
        targets, (scores, _, sum_scores) = forward(model, data, return_hidden=False)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        batch_masks = data[3].numpy()

        for score, target, mask in zip(sub_scores, targets, batch_masks):
            hit_value = float(np.isin(target - 1, score))
            hit.append(hit_value)
            cov.update(score)
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_value = 0.0
            else:
                mrr_value = 1 / (np.where(score == target - 1)[0][0] + 1)
            mrr.append(mrr_value)
            lengths.append(int(np.sum(mask)))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)
    result.append(len(cov))
    print(format_length_bucket_report(summarize_length_buckets(lengths, hit, mrr)))

    return result


def main():
    print('Script started. Initializing seed...')
    init_seed(2024)

    dataset_key = opt.dataset.lower()
    num_node = None

    if dataset_key == 'tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_global = 1
        opt.dropout_gcn = 1
        opt.dropout_local = 0.7
        opt.beta = 0.02
        opt.interests = 5
        opt.length = 8
    elif dataset_key == 'retailrocket':
        num_node = 36969
        opt.n_iter = 1
        opt.dropout_global = 0.5
        opt.dropout_gcn = 0.8
        opt.dropout_local = 0.0
        opt.beta = 0.005
        opt.interests = 3
        opt.length = 12
    elif dataset_key in ('diginetica', 'yoochoose1_64', 'yoochoose1_4'):
        opt.n_iter = 1
        opt.dropout_global = 0.5
        opt.dropout_gcn = 0.8
        opt.dropout_local = 0.0
        opt.beta = 0.005
        opt.interests = 3
        opt.length = 12
    elif dataset_key == 'lastfm':
        num_node = 38616
        opt.n_iter = 1
        opt.dropout_global = 0.1
        opt.dropout_gcn = 0
        opt.dropout_local = 0
        opt.length = 18
        opt.beta = 0.005
        opt.interests = 5
    else:
        raise Exception('Unknown Dataset!')

    print('Loading data...')
    dataset_dir = resolve_dataset_dir(opt.dataset)
    train_data = pickle.load(open(os.path.join(dataset_dir, 'train.txt'), 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(os.path.join(dataset_dir, 'test.txt'), 'rb'))

    if num_node is None:
        all_train_seq = pickle.load(open(os.path.join(dataset_dir, 'all_train_seq.txt'), 'rb'))
        num_node = infer_num_nodes_from_sequences(all_train_seq)

    adj_file = os.path.join(dataset_dir, 'adj_' + str(opt.n_sample_all) + '.pkl')
    num_file = os.path.join(dataset_dir, 'num_' + str(opt.n_sample_all) + '.pkl')

    if not os.path.exists(adj_file):
        print(f"Error: {adj_file} not found! Please run build_graph.py first.")
        return

    adj = pickle.load(open(adj_file, 'rb'))
    num = pickle.load(open(num_file, 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    print('Moving model to CUDA...')
    model = trans_to_cuda(DMIGNN(opt, num_node, adj, num))

    print(opt)
    start = time.time()
    best_result = [0, 0, 0]
    best_epoch = [0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        hit, mrr, cov = train_test(model, train_data, test_data, epoch)
        cov = cov * 100 / (num_node - 1)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        if cov >= best_result[2]:
            best_result[2] = cov
            best_epoch[2] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tCov@20:\t%.4f' % (hit, mrr, cov))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tCov@20:%.4f\t\tEpoch:\t%d,\t%d,\t%d' % (
            best_result[0], best_result[1], best_result[2], best_epoch[0], best_epoch[1], best_epoch[2]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
