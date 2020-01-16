import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_sortie_acf( y_acf, y_len, pacf=False):
    "représentation de la sortie ACF"
    if pacf:
        y_acf = y_acf[1:]
    plt.figure(figsize=(14,6))
    plt.bar(range(len(y_acf)), y_acf, width = 0.1)
    plt.xlabel('lag')
    plt.ylabel('ACF')
    plt.axhline(y=0, color='black')
    plt.axhline(y=-1.96/np.sqrt(y_len), color='b', linestyle='--', linewidth=0.8)
    plt.axhline(y=1.96/np.sqrt(y_len), color='b', linestyle='--', linewidth=0.8)
    plt.ylim(-1, 1)
    plt.show()
    return

def rmse(serie1, serie2):
    return np.sqrt(((serie1-serie2)**2).mean())

def mape(serie1, serie2):
    return (np.abs(1-serie1/serie2)).mean()*100

def ljungbox(sarima_res, retards=[6, 12, 18, 24, 30, 36]):
    return pd.Series(
            {lags: acorr_ljungbox(sarima_res.resid, lags=lags)[1].mean() 
                  for lags in retards
            }
        )

def display_residuals(sarima_result, seuil=0.05):
    
    # shapiro-wilk
    shapiro_pval = stats.shapiro(sarima_result.resid)[1]
    print(f"\nNormalité des résidus - p-valeur SW = {shapiro_pval:.3f} {'X' if shapiro_pval < seuil else 'OK'}", )
    
    # on affiche le Q-Q plot
    fig, ax = plt.subplots(figsize=(2,2))
    _ = st.probplot(sarima_result.resid, plot=ax, fit=True)
    plt.title('Q-Q plot des résidus')
    
    # calcul des retards
    print("\npvaleurs des tests de ljungbox pour les retards:")
    for (retard, pval) in ljungbox(sarima_result).iteritems():
        print(f"{retard}:{pval:.3f}  {'X' if pval < seuil else 'OK'}", end=' | ')