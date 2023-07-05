#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code

import os
import sys
import faiss
import argparse
import torch
import numpy as np

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/tools')
from embed import SentenceEncoder, EncodeLoad, EncodeFile, EmbedLoad
from lib.text_processing import Token, BPEfastApply


###############################################################################
#
# Load texts and remove duplicates
#
###############################################################################

def TextLoadUnify(fname, args):
    if args.verbose:
        print(' - loading texts {:s}: '.format(fname), end='')
    fin = open(fname, encoding=args.encoding, errors='surrogateescape')
    inds = []
    sents = []
    sent2ind = {}
    n = 0
    nu = 0
    for line in fin:
        new_ind = len(sent2ind)
        inds.append(sent2ind.setdefault(line, new_ind))
        if args.unify:
            if inds[-1] == new_ind:
                sents.append(line[:-1])
                nu += 1
        else:
            sents.append(line[:-1])
            nu += 1
        n += 1
    if args.verbose:
        print('{:d} lines, {:d} unique'.format(n, nu))
    del sent2ind
    return inds, sents


###############################################################################
#
# Wrapper for knn on CPU/GPU
#
###############################################################################

def knn(x, y, k, use_gpu):
    return knnGPU(x, y, k) if use_gpu else knnCPU(x, y, k)


###############################################################################
#
# Perform knn on GPU
#
###############################################################################

def knnGPU(x, y, k, mem=5*1024*1024*1024):
    dim = x.shape[1]
    batch_size = mem // (dim*4)
    sim = np.zeros((x.shape[0], k), dtype=np.float32)
    ind = np.zeros((x.shape[0], k), dtype=np.int64)
    for xfrom in range(0, x.shape[0], batch_size):
        xto = min(xfrom + batch_size, x.shape[0])
        bsims, binds = [], []
        for yfrom in range(0, y.shape[0], batch_size):
            yto = min(yfrom + batch_size, y.shape[0])
            # print('{}-{}  ->  {}-{}'.format(xfrom, xto, yfrom, yto))
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(y[yfrom:yto])
            bsim, bind = idx.search(x[xfrom:xto], min(k, yto-yfrom))
            bsims.append(bsim)
            binds.append(bind + yfrom)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(xfrom, xto):
            for j in range(k):
                sim[i, j] = bsims[i-xfrom, aux[i-xfrom, j]]
                ind[i, j] = binds[i-xfrom, aux[i-xfrom, j]]
    return sim, ind

###############################################################################
#
# Index PQ
#
###############################################################################
def knnPQ(x,y,k,code_size):
    dim = x.shape[1]
    idx = faiss.IndexPQ(dim,code_size,8,faiss.METRIC_INNER_PRODUCT)
    print("knnpq")
    #idx = faiss.IndexScalarQuantizer(dim,faiss.ScalarQuantizer.QT_8bit,faiss.METRIC_INNER_PRODUCT)
    print("training")
    idx.train(y)
    idx.add(y)
    sim, ind = idx.search(x, k)
    return sim, ind


###############################################################################
#
# Perform knn on CPU
#
###############################################################################

def knnCPU(x, y, k):
    dim = x.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(y)
    sim, ind = idx.search(x, k)
    return sim, ind


###############################################################################
#
# Scoring
#
###############################################################################

def score(distance, fwd_mean, bwd_mean, margin):
    return margin(distance, (fwd_mean + bwd_mean) / 2)


def score_candidates(candidate_distances, candidate_inds, fwd_mean, bwd_mean, margin, verbose=False):
    if verbose:
        print(' - scoring {:d} candidates'.format(x.shape[0]))
        assert candidate_distances.shape[0]==candidate_inds.shape[0] and candidate_distances.shape[1]==candidate_inds.shape[1]
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] =  score(candidate_distances[i,j], fwd_mean[i], bwd_mean[k], margin)
    return scores


###############################################################################
#
# Main
#
###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LASER: Mine bitext')
    parser.add_argument('src',
        help='Source language corpus')
    parser.add_argument('trg',
        help='Target language corpus')
    parser.add_argument('--encoding', default='utf-8',
        help='Character encoding for input/output')
    parser.add_argument('--src-lang', required=True,
        help='Source language id')
    parser.add_argument('--trg-lang', required=True,
        help='Target language id')
    parser.add_argument('--output', required=True,
        help='Output directory')
    parser.add_argument('--threshold', type=float, default=0,
        help='Threshold on extracted bitexts')

    # mining params
    parser.add_argument('--mode',
        choices=['search', 'score', 'mine'], required=True,
        help='Execution mode')
    parser.add_argument('-k', '--neighborhood',
        type=int, default=4,
        help='Neighborhood size')
    parser.add_argument('--margin',
        choices=['absolute', 'distance', 'ratio'], default='ratio',
        help='Margin function')
    parser.add_argument('--retrieval',
        choices=['fwd', 'bwd', 'max', 'intersect'], default='max',
        help='Retrieval strategy')
    parser.add_argument('--unify', action='store_true',
        help='Unify texts')
    parser.add_argument('--gpu', action='store_true',
        help='Run knn on all available GPUs')
    parser.add_argument('--verbose', action='store_true',
        help='Detailed output')

    # embeddings
    parser.add_argument('--src-embeddings', required=True,
        help='Precomputed source sentence embeddings')
    parser.add_argument('--trg-embeddings', required=True,
        help='Precomputed target sentence embeddings')
    parser.add_argument('--dim', type=int, default=1024,
        help='Embedding dimensionality')
    parser.add_argument('--fp16', action='store_true',
        help='Load precomputed embeddings in float16 format')
    parser.add_argument('--code_size', type=int, default=None,
    help='PQ code size')
    args = parser.parse_args()

    print('LASER: tool to search, score or mine bitexts')
    use_gpu = torch.cuda.is_available() and args.gpu
    if use_gpu:
        print(' - knn will run on all available GPUs (recommended)')
    else:
        print(' - knn will run on CPU (slow)')

    src_inds, src_sents = TextLoadUnify(args.src, args)
    trg_inds, trg_sents = TextLoadUnify(args.trg, args)

    def unique_embeddings(emb, ind, verbose=False):
        aux = {j: i for i, j in enumerate(ind)}
        if verbose:
            print(' - unify embeddings: {:d} -> {:d}'.format(len(emb), len(aux)))
        return emb[[aux[i] for i in range(len(aux))]]

    # load the embeddings and store as np.float32 (required for FAISS)
    x = EmbedLoad(args.src_embeddings, args.dim, verbose=args.verbose, fp16=args.fp16).astype(np.float32)
    if args.unify:
        x = unique_embeddings(x, src_inds, args.verbose)
    faiss.normalize_L2(x)
    y = EmbedLoad(args.trg_embeddings, args.dim, verbose=args.verbose, fp16=args.fp16).astype(np.float32)
    if args.unify:
        y = unique_embeddings(y, trg_inds, args.verbose)
    faiss.normalize_L2(y)
    assert x.shape[0]> args.neighborhood, f" the size of x is {x.shape[0]} and it is less than the k selected {args.neighborhood} "
    assert y.shape[0]> args.neighborhood, f" the size of y is {y.shape[0]} and it is less than the k selected {args.neighborhood} "

    # filenames
    output_prefix = f"sonar.margin_{args.margin}.retrieval_{args.retrieval}.bucc2018.{args.src_lang}-{args.trg_lang}"
    if args.code_size:
        #output_prefix += f".SQ{args.code_size}"
        output_prefix += f".PQ{args.code_size}"
    output_filename = args.output + "/" + output_prefix + f".train.k{args.neighborhood}.candidates.tsv"

    #create results directory:
    directory_name = args.output + "/sim_and_ind"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    common_path = directory_name + "/" +f"{args.src_lang}-{args.trg_lang}"
    if args.code_size:
        common_path+= f".PQ{args.code_size}"
        #common_path+= f".SQ{args.code_size}"
    x2y_ind_file= common_path +".x2y_ind.npy"
    x2y_sim_file =common_path +".x2y_sim.npy"
    y2x_ind_file = common_path +".y2x_ind.npy"
    y2x_sim_file = common_path +".y2x_sim.npy"



    # calculate knn in both directions
    if args.retrieval != 'bwd':
        if not os.path.exists(x2y_ind_file):
            if args.verbose:
                print(' - perform {:d}-nn source against target'.format(args.neighborhood))
            with open(x2y_ind_file,"xb") as f, open(x2y_sim_file,"xb") as g:
                if args.code_size:
                    print("product quantised knn - building:")
                    x2y_sim, x2y_ind = knnPQ(x, y, min(y.shape[0], args.neighborhood),args.code_size)
                else:
                    print("knn - building:")
                    x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], args.neighborhood), use_gpu)
                np.save(f, x2y_ind)
                np.save(g, x2y_sim)

    if args.retrieval != 'fwd':
        if not os.path.exists(y2x_ind_file):
            if args.verbose:
                print(' - perform {:d}-nn target against source'.format(args.neighborhood))
            with open(y2x_ind_file,"xb") as f, open(y2x_sim_file,"xb") as g:
                if args.code_size:
                    print("product quantised knn - building:")
                    y2x_sim, y2x_ind = knnPQ(y, x, min(x.shape[0], args.neighborhood),args.code_size)
                else:
                    print("knn - building:")
                    y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], args.neighborhood), use_gpu)
                np.save(f, y2x_ind)
                np.save(g, y2x_sim)

    # margin function
    if args.margin == 'absolute':
        margin = lambda a, b: a
    elif args.margin == 'distance':
        margin = lambda a, b: a - b
    else:  # args.margin == 'ratio':
        margin = lambda a, b: a / b
    assert not os.path.exists(output_filename), f"the candidates file {output_filename} exists, no need to re-run."
    fout = open(output_filename, mode='w', encoding=args.encoding, errors='surrogateescape')

    #if args.retrieval != 'bwd':
    x2y_ind = np.load(x2y_ind_file)
    x2y_sim = np.load(x2y_sim_file)

    #if args.retrieval != 'fwd':
    y2x_ind = np.load(y2x_ind_file)
    y2x_sim = np.load(y2x_sim_file)

    # select subset of relevant neighbours
    if args.neighborhood < y2x_sim.shape[1]:
        print(f"loading bigger file and adjusting size to {args.neighborhood}:")
        x2y_ind = x2y_ind[:,:args.neighborhood]
        x2y_sim = x2y_sim[:,:args.neighborhood]
        y2x_ind = y2x_ind[:,:args.neighborhood]
        y2x_sim = y2x_sim[:,:args.neighborhood]

    #denominators
    x2y_mean = x2y_sim.mean(axis=1)
    y2x_mean = y2x_sim.mean(axis=1)

    if args.mode == 'search':
        if args.verbose:
            print(' - Searching for closest sentences in target')
            print(' - writing alignments to {:s}'.format(output_filename))

        scores = score_candidates(x2y_sim, x2y_ind, x2y_mean, y2x_mean, margin, args.verbose)
        best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

        nbex = x.shape[0]
        ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
        err = nbex - np.equal(best.reshape(nbex), ref).astype(int).sum()
        print(' - errors: {:d}={:.2f}%'.format(err, 100*err/nbex))
        for i in src_inds:
            print(trg_sents[best[i]], file=fout)

    elif args.mode == 'score':
        for i, j in zip(src_inds, trg_inds):
            s = score(x2y_sim[i,j], x2y_mean[i], y2x_mean[j], margin)
            print(s, src_sents[i], trg_sents[j], sep='\t', file=fout)

    elif args.mode == 'mine':
        if args.verbose:
            print(' - mining for parallel data')

        fwd_scores = score_candidates(x2y_sim, x2y_ind, x2y_mean, y2x_mean, margin, args.verbose)
        bwd_scores = score_candidates(y2x_sim, y2x_ind, y2x_mean, x2y_mean, margin, args.verbose)

        fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
        bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]
        if args.verbose:
            print(' - writing alignments to {:s}'.format(output_filename))
            if args.threshold > 0:
                print(' - with threshold of {:f}'.format(args.threshold))
        if args.retrieval == 'fwd':
            for i, j in enumerate(fwd_best):
                print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
        if args.retrieval == 'bwd':
            for j, i in enumerate(bwd_best):
                print(bwd_scores[j].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
        if args.retrieval == 'intersect':
            for i, j in enumerate(fwd_best):
                if bwd_best[j] == i:
                    print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
        if args.retrieval == 'max':
            indices = np.stack((np.concatenate((np.arange(x.shape[0]), bwd_best)),
                                np.concatenate((fwd_best, np.arange(y.shape[0])))), axis=1)
            scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
            seen_src, seen_trg = set(), set()
            for i in np.argsort(-scores):
                src_ind, trg_ind = indices[i]
                if not src_ind in seen_src and not trg_ind in seen_trg:
                    seen_src.add(src_ind)
                    seen_trg.add(trg_ind)
                    if scores[i] > args.threshold:
                        print(scores[i], src_sents[src_ind], trg_sents[trg_ind], sep='\t', file=fout)

    fout.close()
