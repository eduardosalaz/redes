#from sklearn.preprocessing import CategoricalEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = np.loadtxt(fname="MuestraFallas3.txt", skiprows=1,   dtype='str', usecols=(0), unpack=True)
nrows = data.shape[0]
print 'Numero de renglones en la tabla de datos:', nrows
fallasdata = {}
for i in range(6):
      if (i == 0) or (i==1) or (i==4) or (i==6):
         typ = 'str'
         data = np.loadtxt(fname="MuestraFallas3.txt", skiprows=1,   dtype=typ, usecols=(i), unpack=True)
         label_enc = LabelEncoder()
         label_enc.fit(data)
         data_encoded = label_enc.transform(data)
         data = np.asarray(data_encoded)
         fallasdata[i] = data
      else:
          typ = 'int'
          data = np.loadtxt(fname="MuestraFallas3.txt", skiprows=1,   dtype=typ, usecols=(i), unpack=True)
          fallasdata[i] = data
fallasdata = pd.DataFrame.from_dict(fallasdata)
fallasdata.columns=['zona','falla','mes','an-o','lote','marca']
fallasdata.to_csv('fallasdata.csv')
  
print 'Correlacion entre fallas con marca:', fallasdata['falla'].corr(fallasdata['marca'])
print 'Correlacion entre fallas con lote:', fallasdata['falla'].corr(fallasdata['lote'])
print 'Correlacion entre fallas con zona:', fallasdata['falla'].corr(fallasdata['zona'])
print 'Correlacion entre fallas con mes:', fallasdata['falla'].corr(fallasdata['mes'])
print 'Correlacion entre fallas con a-o:', fallasdata['falla'].corr(fallasdata['an-o'])


          
            
      
