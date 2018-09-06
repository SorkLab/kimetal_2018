#! /usr/bin/env python

"""
usage:
dtest.py P1,P2,P3,OUTGROUP

in the PX positions choose from the following clade abbreviations:
    QBN, QBS, QDD, QDG, QDU, QP, QJT, QDP, QCM, QL, QE, QA, B, CM, SCRUB
outgroup should be QK

vcf input file has sample IDs as column names in the header row. if they do not 
match up to the IDs in the species dictionary below this script will not work.

e.g. python dtest.py QBN,QP,QE,QK
"""

import sys
import random
import numpy
import scipy.stats
import itertools

species={
    "QBN":("QB-59.1","QB-66b.8","QB-34.5","QB-20b.2","QB-20b.5"),
    "QBS":("QB-18.6","QB-PR.315","QB-NEW.3","QB-NEW.4","QB-NEW.5",
    	   "QB-SD.1","QB-SD.3","QB-SD.4","QB-12.1","QB-10.3"),
    "QDD":("QDD-37b.2","QDD-48.2","QDD-48.7","QDD-50b.1","QDD-62.6","QDD-62.4"),
    "QDG":("QDG-45b.9","QDG-45b.10","QDG-MtB.1","QDG-MtB.8"),
    "QDU":("QDU-19.2","QDU-SBBG.1"),
    "QP":("QP-SC3.20","QP-Catb.19","QP-Cat.4","QP-SR33.3","QP-SR.7"),
    "QJT":("QJT-22.3","QJT-21b.6","QJT-21b.4"),
    "QDO":("QDO-SW.1","QDO-SW.10","QDO-SW.4"),
    "QCM":("QCM-9.1","QCM-11.7","QCM-16.4","QCM-17b.11"),
    "QL":("QL-PAY.6","QL-SW.786","QL-UCD.8"),
    "QE":("QE-Ju.161","QE-17.1","QE-15.2"),
    "QK":("QK-PM316","QK-PM266"),
    "QA":("QA-204","QP-SC3.20","QP-Catb.19","QP-Cat.4","QP-SR33.3","QP-SR.7",
    	  "QJT-22.3","QJT-21b.6","QJT-21b.4","QDO-SW.1","QDO-SW.10","QDO-SW.4",
    	  "QJT-22.3","QJT-21b.6","QJT-21b.4","QDO-SW.1","QDO-SW.10","QDO-SW.4",
    	  "QCM-9.1","QCM-11.7","QCM-16.4","QCM-17b.11"),
    "B":("QB-59.1","QB-66b.8","QB-34.5","QB-20b.2","QB-20b.5","QB-18.6",
         "QB-PR.315","QB-NEW.3","QB-NEW.4","QB-NEW.5","QB-SD.1","QB-SD.3",
         "QB-SD.4","QB-12.1","QB-10.3","QDD-37b.2","QDD-48.2","QDD-48.7",
         "QDD-50b.1","QDD-62.6","QDD-62.4","QDG-45b.9","QDG-45b.10",
         "QDG-MtB.1","QDG-MtB.8"),
    "CM":("QDU-19.2","QDU-SBBG.1","QDU-19.2","QDU-SBBG.1","QP-SC3.20",
         "QP-Catb.19","QP-Cat.4","QP-SR33.3","QP-SR.7","QJT-22.3","QJT-21b.6",
         "QJT-21b.4","QDO-SW.1","QDO-SW.10","QDO-SW.4","QCM-9.1","QCM-11.7",
         "QCM-16.4","QCM-17b.11"),
    "SCRUB":("QB-59.1","QB-66b.8","QB-34.5","QB-20b.2","QB-20b.5","QB-18.6",
          "QB-PR.315","QB-NEW.3","QB-NEW.4","QB-NEW.5","QB-SD.1","QB-SD.3",
          "QB-SD.4","QB-12.1","QB-10.3","QDD-37b.2","QDD-48.2","QDD-48.7",
          "QDD-50b.1","QDD-62.6","QDD-62.4","QDG-45b.9","QDG-45b.10",
          "QDG-MtB.1","QDG-MtB.8")
}

def comp_af(gts):
    return float('/'.join(gts).count('1'))/(2*len(gts))

def compute_d(af_list):
    one, two, three, four = zip(*af_list)
    nums = -numpy.multiply(
                           numpy.subtract(three, four),
                           numpy.subtract(one,two)
    )
    denoms = numpy.multiply(
                            numpy.subtract(numpy.add(three,four),
                                           numpy.multiply(three,four)*2),
                            numpy.subtract(numpy.add(one,two),
                                           numpy.multiply(one,two)*2)
    )
    d = sum(nums)/sum(denoms)
    return d

def compute_d_single(af_list):
    one, two, three, four = zip(*af_list)
    nums = -numpy.multiply(
                           numpy.subtract(three, four),
                           numpy.subtract(one,two)
    )
    denoms = numpy.multiply(
                            numpy.subtract(numpy.add(three,four),
                                           numpy.multiply(three,four)*2),
                            numpy.subtract(numpy.add(one,two),
                                           numpy.multiply(one,two)*2)
    )
    d = sum(nums)/sum(denoms)
    return (d,sum(denoms))

def jackknife_estimator2(data,mjs,sampest):
    g = len(data)
    n = sum(mjs)
    estimean = numpy.multiply(g,sampest)
    subsetests = []
    for i,mj in zip(range(0, g),mjs):
        data_subset = numpy.delete(data, i, 0)
        data_subsetest = compute_d(numpy.concatenate(data_subset))
        subsetests.append(data_subsetest)  
        estimean = numpy.add(estimean,-numpy.multiply((1-mj/float(n)),
            data_subsetest))
    adj_resids = []
    for i,mj,subsetest in zip(range(0,g),mjs,subsetests):
        h = float(n)/mj
        theta_tilde = (numpy.multiply(h, sampest) 
                       - numpy.multiply((h-1),subsetest))
        resid = numpy.add(theta_tilde,-estimean) ** 2
        adj_resid = numpy.multiply(resid,(1/(h-1)))
        adj_resids.append(adj_resid)
    sigma_hat= numpy.sqrt(numpy.multiply(1/float(g),
        numpy.sum(numpy.array(adj_resids),0)))
    return [estimean, sigma_hat]

def main():
    one, two, three, four = sys.argv[1].split(',')
    
    infile = open('data.vcf','rU')
    lines = infile.readlines()
    
    outfilename = '{0}_{1}_{2}_{3}.csv'.format(one, two, three, four)
    outfile = open(outfilename ,'w')
    outfile.write(
        'one,two,three,four,Dhat,sigmahat,Z,p,snpcount,total_abbababa\n')
    outfile.close()
    
    chunknum=500
    
    indivsstring = '{0};{1};{2};{3}'.format(one, two, three, four)
    one_list = species[one]
    two_list = species[two]
    three_list = species[three]
    four_list = species[four]
    af_list = []
    
    print 'P1:{0},P2:{1},P3:{2},OUT:{3}'.format(one, two, three, four)
    
    for line in lines:
        if line.startswith('##'):
            continue
        elif line.startswith('#'):
            line = line.strip('\n').split('\t')
            one_index = [line.index(x) for x in one_list]
            two_index = [line.index(x) for x in two_list]
            three_index = [line.index(x) for x in three_list]
            four_index = [line.index(x) for x in four_list]
            indices = one_index + two_index + three_index + four_index
        else:
            line = line.strip('\n').split('\t')
            if (len(line[4]) == 1) and (len(line[3]) == 1):
                gts = [line[x][0:3] for x in indices]
                checkmissing = (len(gts) == sum(['.' not in x for x in gts]))
                if checkmissing:
                    afs = [
                        comp_af(gts[0:len(one_index)]),
                        comp_af(gts[len(one_index):
                                len(one_index)+len(two_index)]),
                        comp_af(gts[len(one_index)+len(two_index):
                                len(one_index)+len(two_index)
                                +len(three_index)]), 
                        comp_af(gts[-len(four_index):])
                    ]
                    af_list.append(afs)
    
    D, sumabbababa = compute_d_single(af_list)
    data_chunks = numpy.array_split(af_list, chunknum)
    snpcounts = [len(x) for x in data_chunks]
    jackknife_D, jackknife_sigma = jackknife_estimator2(data_chunks, 
                                                        snpcounts, D)
    jackknife_Z = jackknife_D/jackknife_sigma
    jackknife_p = scipy.stats.norm.sf(abs(jackknife_Z))*2
    
    outvals = [one, two, three, four, jackknife_D, jackknife_sigma,
               jackknife_Z, jackknife_p, sum(snpcounts), sumabbababa]
    outline = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'.format(*outvals)
    outfile = open(outfilename,'a')
    outfile.write(outline)
    outfile.close()

if __name__ == "__main__":
    main()