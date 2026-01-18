import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_korean_font():
    '''
    set_korean_font

    그래프에서 한글 깨짐 방지 설정
    '''
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    print('한글 폰트 설정 완료')

def save_plot(filename):
    '''
    save_plot

    그래프를 'plots' 폴더에 저장하는 함수
    
    :param filename: plot을 저장할 경로
    '''
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{filename}.png', bbox_inches = 'tight')
    print(f'그래프가 plots/{filename}.png에 저장되었습니다.')

def draw_kdeplot(df, var_name, hue_name):
    '''
    draw_kdeplot의 Docstring
    
    :param df: 데이터(DataFrame)
    :param var_name: 수치형 변수 이름(string)
    :param hue_name: 범주형 변수 이름(string)
    '''

    plt.figure(figsize = (10,6))

    sns.kdeplot(data = df, x = var_name, hue = hue_name, shade = True)
    plt.title(f'{var_name}에 대한 {hue_name} 분포')
    plt.show()

# 예시 보여드리기 위해 gemini 보고 만들라고 했습니당
def automate_impute(df):
    """
    데이터프레임의 결측치를 자동으로 대체합니다.
    - 숫자형: 중앙값(median)으로 대체 (이상치 영향 최소화)
    - 범주형(문자형): 최빈값(mode)으로 대체
    """
    df_clean = df.copy()
    
    # 1. 숫자형 칼럼 처리
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"  - [숫자형] {col}: 중앙값({median_val})으로 대체 완료")

    # 2. 범주형(문자형/Object) 칼럼 처리
    obj_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in obj_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"  - [범주형] {col}: 최빈값('{mode_val}')으로 대체 완료")
            
    return df_clean