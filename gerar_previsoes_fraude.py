#Importação de pacotes
def gerar_previsoes_modelo():
    
    print("Preparando ambiente...")
    
    
    import pandas as pd    
    import joblib
    from sklearn.ensemble import RandomForestClassifier    
    from sklearn.model_selection import train_test_split    
    import numpy as np
    from sklearn.preprocessing import LabelEncoder #Utilizada para fazer o OneHotEncoding
    from sklearn.metrics import mean_squared_error,precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
    from imblearn import under_sampling, over_sampling #Utilizada para fazer o balanceamento de dados
    from imblearn.over_sampling import SMOTE #Utilizada para fazer o balanceamento de dados
    from sklearn.preprocessing import MinMaxScaler #Utilizada para fazer a padronização dos dados
    from sklearn.metrics import r2_score # Utilizado para medir a acuracia do modelo preditivo    
    import pandas as pd
    
    
    clf = joblib.load('modelo_treinado_fraude.pk')
    
    
    
    print("Gerando Previsões...")

    df_original = pd.read_csv("novos_dados.csv")
    
    df_original['Estado_Civil'] = df_original['Estado_Civil'].replace(['NENHUM'], 'OUTRO')
    df_original['Estado_Civil'] = df_original['Estado_Civil'].replace(['UNIÃO ESTAVEL'], 'CASADO (A)')

    bins = [0, 21, 30, 40, 50, 60, 100]
    labels = ['Até 21 Anos', 'De 22 até 30 Anos', 'De 31 até 40 Anos', 'De 41 até 50 Anos', 'De 51 até 60', 'Acima de 60 Anos']
    df_original['Faixa_Etaria'] = pd.cut(df_original['Idade'], bins=bins, labels=labels)
    
    bins = [-100, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 9000000000]
    labels = ['Até 1k', 'De 1k até 2k', 'De 2k até 3k', 'De 3k até 5k', 'De 5k até 10k', 'De 10k até 20k',
          'De 20k até 30k', 'Acima de 50k']
    df_original['Faixa_Salarial'] = pd.cut(df_original['Valor_Renda'], bins=bins, labels=labels)

    df_original['QT_Dias_Atraso'] = df_original['QT_Dias_Atraso'].fillna((df_original['QT_Dias_Atraso'].median()))

    bins = [-100, 30, 60, 90, 180, 240, 360, 500]
    labels = ['Até 30 dias', 'De 31 até 60', 'De 61 até 90', 'De 91 até 180', 'De 181 até 240','De 241 até 360', 'Acima de 360']
    df_original['Faixa_Dias_Atraso'] = pd.cut(df_original['QT_Dias_Atraso'], bins=bins, labels=labels)
    

    bins = [0, 60, 120, 200, 720]
    labels = ['Até 60 Meses', 'De 61 até 120 Meses', 'De 121 até 200 Meses', 'Acima de 200 Meses']
    df_original['Faixa_Prazo_Emprestimo'] = pd.cut(df_original['Prazo_Emprestimo'], bins=bins, labels=labels)
    

    bins = [-1, 60, 120, 200, 500]
    labels = ['Até 60 Meses', 'De 61 até 120 Meses', 'De 121 até 200 Meses', 'Acima de 200 Meses']
    df_original['Faixa_Prazo_Restante'] = pd.cut(df_original['Prazo_Restante'], bins=bins, labels=labels)
    
    

    columns = ['Sexo', 'UF_Cliente', 'Perc_Juros', 
       'VL_Emprestimo', 'VL_Emprestimo_ComJuros', 'QT_Total_Parcelas_Pagas',
       'QT_Total_Parcelas_Pagas_EmDia', 'QT_Total_Parcelas_Pagas_EmAtraso',
       'Qt_Renegociacao', 'Estado_Civil', 'QT_Parcelas_Atraso', 'Saldo_Devedor', 
       'Total_Pago', 'Faixa_Prazo_Restante', 'Faixa_Salarial', 'Faixa_Prazo_Emprestimo', 'Faixa_Etaria', 
       'Faixa_Dias_Atraso', 'Possivel_Fraude']

    df_dados = pd.DataFrame(df_original, columns=columns)


    variaveis_categoricas = []
    for i in df_dados.columns[0:18].tolist():
        if df_dados.dtypes[i] == 'object' or df_dados.dtypes[i] == 'category':                        
            variaveis_categoricas.append(i)  

    
    
    lb = LabelEncoder()
    for var in variaveis_categoricas:
        df_dados[var] = lb.fit_transform(df_dados[var])

    for var in variaveis_categoricas:
        df_dados[var] = lb.fit_transform(df_dados[var])

    PREDITORAS = df_dados.iloc[:, 0:18]

    Normalizador = MinMaxScaler()
    dados_normalizados = Normalizador.fit_transform(PREDITORAS)    


    previsoes = clf.predict(dados_normalizados)
    probabilidades = clf.predict_proba(dados_normalizados)
    df_original['PREVISOES'] = previsoes
    df_original['PROBABILIDADES'] = probabilidades[:, 1]       

    df_original.to_excel('previsoes_fraude.xlsx')
    print("Previsoes Geradas com Sucesso!")        


def main():
    gerar_previsoes_modelo()    


if __name__ == "__main__":
    main()    

