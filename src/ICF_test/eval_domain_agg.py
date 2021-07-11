"""
@Author StellaVerkijk
This scripts writes the performance of a model for ICF classification on note level to a file
"""


def eval_per_domain(dict_predict, dict_ann, outfilename):
    """
    calculates precision, recall and f1 per class
    dict_predict & dict_ann are dictionaries of the form {note_id : ['label1', 'label2'], note_id: ['label']}
    
    """
    tp_none = 0
    fp_none = 0
    fn_none = 0

    tp_lopen = 0
    fp_lopen = 0
    fn_lopen = 0

    tp_stemming = 0
    fp_stemming = 0
    fn_stemming = 0

    tp_ber = 0
    fp_ber = 0
    fn_ber = 0

    tp_ins = 0
    fp_ins = 0
    fn_ins = 0

    for key, value in dict_predict.items():
        if 'None' in value and 'None' in dict_ann[key]:
            tp_none += 1
        if 'None' in value and 'None' not in dict_ann[key]:
            fp_none +=1
        if 'None' not in value and 'None' in dict_ann[key]:
            fn_none +=1
        if '.D450: Lopen en zich verplaatsen' in value and '.D450: Lopen en zich verplaatsen' in dict_ann[key]:
            tp_lopen += 1
        if '.D450: Lopen en zich verplaatsen' in value and '.D450: Lopen en zich verplaatsen' not in dict_ann[key]:
            fp_lopen +=1
        if '.D450: Lopen en zich verplaatsen' not in value and '.D450: Lopen en zich verplaatsen' in dict_ann[key]:
            fn_lopen +=1
        if '.B152: Stemming' in value and '.B152: Stemming' in dict_ann[key]:
            tp_stemming += 1
        if '.B152: Stemming' in value and '.B152: Stemming' not in dict_ann[key]:
            fp_stemming +=1
        if '.B152: Stemming' not in value and '.B152: Stemming' in dict_ann[key]:
            fn_stemming +=1
        if '.D840-859: Beroep en werk' in value and '.D840-859: Beroep en werk' in dict_ann[key]:
            tp_ber += 1
        if '.D840-859: Beroep en werk' in value and '.D840-859: Beroep en werk' not in dict_ann[key]:
            fp_ber +=1
        if '.D840-859: Beroep en werk' not in value and '.D840-859: Beroep en werk' in dict_ann[key]:
            fn_ber +=1
        if '.B455: Inspanningstolerantie' in value and '.B455: Inspanningstolerantie' in dict_ann[key]:
            tp_ins += 1
        if '.B455: Inspanningstolerantie' in value and '.B455: Inspanningstolerantie' not in dict_ann[key]:
            fp_ins +=1
        if '.B455: Inspanningstolerantie' not in value and '.B455: Inspanningstolerantie' in dict_ann[key]:
            fn_ins +=1
            
    precision_none = tp_none / (tp_none + fp_none)
    recall_none = tp_none / (tp_none + fn_none) 
    f1_none =  (2 * (precision_none * recall_none)) / (precision_none + recall_none)

    precision_lopen = tp_lopen / (tp_lopen + fp_lopen)
    recall_lopen = tp_lopen / (tp_lopen + fn_lopen)
    f1_lopen = (2 * (precision_lopen * recall_lopen)) / (precision_lopen + recall_lopen)

    precision_stem = tp_stemming / (tp_stemming + fp_stemming)
    recall_stem = tp_stemming / (tp_stemming + fn_stemming)
    f1_stem = (2 * (precision_stem * recall_stem)) / (precision_stem + recall_stem)
    
    if tp_ber + fp_ber != 0:
        precision_ber = tp_ber / (tp_ber + fp_ber)
    else: 
        precision_ber = 0
    if tp_ber + fn_ber != 0:
        recall_ber = tp_ber / (tp_ber + fn_ber)
    else:
        recall_ber = 0
    if precision_ber + recall_ber != 0:
        f1_ber = (2 * (precision_ber * recall_ber)) / (precision_ber + recall_ber)
    else:
        f1_ber = 0

    precision_ins = tp_ins / (tp_ins + fp_ins)
    recall_ins = tp_ins / (tp_ins + fn_ins)
    f1_ins = (2 * (precision_ins * recall_ins)) / (precision_ins + recall_ins)
    
    
    with open(outfilename, "w+", encoding="utf-8") as outfile:
        outfile.write(" \tprecision\trecall\tf1\n")
        outfile.write(f"lopen\t{precision_lopen}\t{recall_lopen}\t{f1_lopen}\n")
        outfile.write(f"stemming\t{precision_stem}\t{recall_stem}\t{f1_stem}\n")
        outfile.write(f"beroep\t{precision_ber}\t{recall_ber}\t{f1_ber}\n")
        outfile.write(f"inspanningst\t{precision_ins}\t{recall_ins}\t{f1_ins}\n")
