import shutil, os


for usina in os.listdir('Usinas'):
    path_raw = os.path.join('Usinas', usina, 'Minimização')
    path1 = os.path.join(path_raw, 'Padrão')
    path2 = os.path.join(path_raw, 'Regressão L2')
    
    shutil.rmtree(os.path.join(path_raw, 'Gráficos'))
    shutil.rmtree(os.path.join(path1, 'Gráficos'))
    shutil.rmtree(os.path.join(path2, 'Gráficos'))