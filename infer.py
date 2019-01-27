from tensorflow.keras.models import load_model as l
from numpy import argmax as g, expand_dims as d, load
class Z():
 def __init__(a):
  a.i=load("w")
  a.s=[len(a.i)+1]*10
  a.w={v:k for k,v in a.i.items()}
  a.m=l("b")
 def p(a,n):
  a.s.pop(0)
  a.s.append(a.w[n])
  return a.i[g(a.m.predict(d(d(a.s,0),2),1))]

if __name__ == "__main__":
 from tqdm import tqdm
 t = Z() 
 errors = 0
 with open("mobydick.txt", 'r') as myfile:
  with open('prediction.txt', 'w') as out:
   data = myfile.read()
   data_compare = zip(data[:-1], data[:-1]) 
    #    print(list(data_compare))
#    for inp,gt in tqdm(data_compare, total=len(data[:-1])):
   for inp,gt in data_compare:
    pred = t.p(inp)
    print("{} : {}".format(gt,pred))
    if pred != gt:
     errors += 1
    out.write(pred)
   print(errors)

#  incorrect = 0
#  t = Z() 
#  with open("mobydick.txt", 'r') as file:
#   p_ch = ch = file.read(1)
#   while True:
#    ch = file.read(1)
#    if not ch:
#     break
#    f_ch = t.p(p_ch)
#    if f_ch != ch:
#     incorrect += 1
#    p_ch = ch

#  print(incorrect)