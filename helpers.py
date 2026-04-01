import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def convert_to_numeric(X):
    X["sex"] = X["sex"].map({"male": 1, "female": 0})
    X["smoker"] = X["smoker"].map({"yes": 1, "no": 0})
    X["region"] = X["region"].map({"southwest": 1, "southeast": 2, "northwest": 3, "northeast": 4})
    return X

def draw_importance(best_model) -> None:
    importance = best_model.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values(by='Importance', ascending=False)

    top_n = 6
    plt.figure(figsize=(10, 8))
    plt.barh(
        importance_df['Feature'].head(top_n)[::-1],
        importance_df['Importance'].head(top_n)[::-1],
        color='skyblue'
    )
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np

def inject_missing_values(df, missing_rate=0.1):
    random_matrix = np.random.rand(*df.shape)
    
    mask = random_matrix < missing_rate
    
    df_noisy = df.mask(mask, np.nan)
    
    return df_noisy
