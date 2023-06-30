import glob
import os
import string
import textgrid
import re
import argparse
parser = argparse.ArgumentParser(description="Prepare L2 data")
parser.add_argument("--l2_path",default="/home/v-wenxuanye/data/L2_ARTIC_v5",help="l2-Arctic path")
parser.add_argument("--save_path",default="./data/l2_test",help="l2-Arctic path")
parser.add_argument("--use_l2_phone_set", default=True, help="whether to use l2 phone set labelled in version 5")

args = parser.parse_args()

path = args.l2_path+"/*/annotation/*.TextGrid"
#   Spanish\Vietnamese\Hindi\Mandarin\Korean\Arabic
train_spk = ["MBMPS","ERMS","TLV","PNV","ASI","RRBI","TXHC","LXC","HJK","HKK","ABA","SKA"]
dev_spk = ["EBVS","THV","TNI","BWC","YDCK","YBAA"]
test_spk = ["NJS","HQTV","SVBI","NCC","YKWK","ZHAA"] 
load_error_file = ["YDCK/annotation/arctic_a0209.TextGrid",
                  "YDCK/annotation/arctic_a0272.TextGrid"]
wav_lst = glob.glob(path)
tmp_path = args.save_path
# os.system("mkdir %s" % tmp_path)
os.makedirs(tmp_path, exist_ok=True)
type_ = args.save_path.split("_")
w = open(tmp_path+"/wrd_text",'w+')
w1 = open(tmp_path+"/wav.scp",'w+')
w2 = open(tmp_path+"/wav_sph.scp",'w+')
w3 = open(tmp_path+"/phn_text",'w+')
w4 = open(tmp_path+"/transcript_phn_text",'w+')

remove_digits = str.maketrans('', '', string.digits)
remove_underline = str.maketrans('', '', '_')
remove_star = str.maketrans('', '', '*')

def del_repeat_sil(phn_lst):
    tmp = [phn_lst[0]]
    for i in range(1,len(phn_lst)):
        if(phn_lst[i] == phn_lst[i-1] and phn_lst[i]=="sil"):
            continue
        else:
            tmp.append(phn_lst[i])
    return tmp

for phn_path in wav_lst:
    if(  "/".join(phn_path.split("/")[-3:]) in load_error_file  ):
        continue
    spk_id = phn_path.split("/")[-3]
    utt_id = spk_id + "_" + phn_path.split("/")[-1][:-9]
    tmp = re.sub("annotation","wav",phn_path)
    wav_path = re.sub("TextGrid","wav",tmp)
    tmp = re.sub("annotation","transcript",phn_path)
    text_path = re.sub("TextGrid","txt",tmp)
    if(spk_id in eval(type_[-1]+"_spk") ):
        cur_phns = []
        transcript_phns = ['sil']
        tg = textgrid.TextGrid.fromFile(phn_path)
        end_of_words = [tg[0][i].maxTime for i in range(len(tg[0])) if tg[0][i].mark != '']
        for i in tg[1]:
            if(i.mark == ''):
                # transcript_phns.append(("sil"))
                cur_phns.append("sil")
            else:
                trans_human_type = i.mark.split(",")
                if( len(trans_human_type) == 1 ):
                    phn = trans_human_type[0]
                    trans_phn = trans_human_type[0].translate(remove_digits)
                else:
                    if trans_human_type[2].strip() == 'a':
                        # insertion error
                        trans_phn = ''                        
                        phn = trans_human_type[1]
                    elif trans_human_type[2].strip() == 'd':
                        # deletion error
                        trans_phn = trans_human_type[0].translate(remove_digits)
                        phn = ''
                    else:
                        trans_phn = trans_human_type[0].translate(remove_digits)
                        phn = trans_human_type[1]
                
                ## phn 
                if phn != '':
                    if args.use_l2_phone_set:
                        phn = phn.translate(remove_digits).translate(remove_underline)
                    else:
                        phn = phn.translate(remove_digits).translate(remove_underline).translate(remove_star)
                    
                    if(phn == "sp" or phn == "SIL" or phn == " " or phn == "spn" or phn == 'sil' ):
                        cur_phns.append("sil")
                    else:
                        phn = phn.strip(" ")
                        if(phn == "ERR" or phn == "err"):
                            cur_phns.append("err")
                        elif(phn == "ER)"):
                            cur_phns.append("er")
                        elif(phn == "AX" or phn == "ax" or phn == "AH)"):  # split ax and ah
                            cur_phns.append("ah")
                        elif(phn == "V``"):
                            cur_phns.append("v")
                        elif(phn == "W`"):
                            cur_phns.append("w")
                        else:
                            cur_phns.append(phn.lower())
                        
                ## trans phn 
                if trans_phn != '':
                    if(trans_phn == "sp" or trans_phn == "SIL" or trans_phn == " " or trans_phn == "spn" or trans_phn == "sil"):
                        # transcript_phns.append(("sil"))
                        pass
                    else:
                        trans_phn = trans_phn.strip(" ")
                        if(trans_phn == "ERR" or trans_phn == "err"):
                            transcript_phns.append("err")
                        elif(trans_phn == "ER)"):
                            transcript_phns.append("er")
                        elif(trans_phn == "AX" or trans_phn == "ax" or trans_phn == "AH)"):  # split ax and ah
                            transcript_phns.append("ah")
                        elif(trans_phn == "V``"):
                            transcript_phns.append("v")
                        elif(trans_phn == "W`"):
                            transcript_phns.append("w")    
                        else:
                            transcript_phns.append(trans_phn.lower())
            if i.maxTime in end_of_words:
                transcript_phns.append("sil")
                        
                        
        f = open(text_path,'r')
        for line in f:
            w.write(utt_id + " " + line.lower() + "\n")
        w1.write(utt_id + " " + wav_path + "\n" )
        w2.write(utt_id + " " + wav_path + "\n" )
        w3.write(utt_id + " " + " ".join(cur_phns) + "\n" )
        w4.write(utt_id + " " + " ".join(transcript_phns) + "\n" )
        # w3.write(utt_id + " " + " ".join(del_repeat_sil(cur_phns)) + "\n" )
        # w4.write(utt_id + " " + " ".join(del_repeat_sil(transcript_phns)) + "\n" )
        
    
w.close()
w1.close()
w2.close()
w3.close()
w4.close()
