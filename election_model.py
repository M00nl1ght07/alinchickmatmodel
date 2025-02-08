import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class ElectionPredictor:
    def __init__(self):
        # Загрузка данных
        self.data = pd.read_excel('data.xlsx')
        self.party_names = pd.read_excel('party_names.xlsx')
        
        # Очистка и подготовка данных
        self.clean_data()
        self.prepare_data()
        self.train_model()
        self.visualize_results()
    
    def clean_data(self):
        # Список числовых столбцов
        numeric_columns = ['Population', 'Population_urban', 'Population_rural', 
                         'Poverty', 'Unemployment', 'Employment', 'GDP', 
                         'Inflation', 'Average_salary']
        
        # Очистка данных
        for col in numeric_columns:
            # Заменяем тире на NaN
            self.data[col] = self.data[col].replace('–', np.nan)
            # Заменяем запятые на точки
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].apply(lambda x: str(x).replace(',', '.') if pd.notna(x) else x)
            # Конвертируем в float
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Также очищаем столбцы с процентами голосов
        for i in range(1, 15):
            col = f'Party_{i}_percent'
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
    
    def prepare_data(self):
        # Выбираем только строки с результатами выборов (2011, 2016, 2021)
        election_years = [2011, 2016, 2021]
        self.election_data = self.data[self.data['Year'].isin(election_years)].copy()
        
        # Признаки для модели
        self.features = ['Population', 'Population_urban', 'Population_rural', 
                        'Poverty', 'Unemployment', 'Employment', 'GDP', 
                        'Inflation', 'Average_salary']
        
        # Заполняем пропущенные значения средними по году
        for feature in self.features:
            mean_values = self.election_data.groupby('Year')[feature].transform('mean')
            self.election_data[feature] = self.election_data[feature].fillna(mean_values)
        
        # Проверяем, что все данные корректно преобразованы
        print("\nПроверка данных после подготовки:")
        print(f"Количество строк: {len(self.election_data)}")
        print("\nТипы данных:")
        print(self.election_data[self.features].dtypes)
        print("\nПроверка на пропущенные значения:")
        print(self.election_data[self.features].isnull().sum())
    
    def train_model(self):
        self.models = {}
        self.scores = {}
        
        print("Обучение моделей:")
        # Обучаем модель для каждой партии
        for year in [2011, 2016, 2021]:
            year_data = self.party_names[self.party_names['Year'] == year]
            for _, party_info in year_data.iterrows():
                party_id = party_info['Party_N']
                party_name = party_info['Party_name']
                party_col = f'Party_{party_id}_percent'
                
                if party_col in self.election_data.columns:
                    # Берем только строки для текущего года
                    year_election_data = self.election_data[self.election_data['Year'] == year]
                    X = year_election_data[self.features]
                    y = year_election_data[party_col]
                    
                    # Убираем строки с пропущенными значениями
                    mask = y.notna()
                    X = X[mask]
                    y = y[mask]
                    
                    if len(y) > 0:
                        print(f"Год {year}, Партия {party_name}: {len(y)} наблюдений")
                        # Разделяем данные на обучающую и тестовую выборки
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Обучаем модель
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Сохраняем модель и оценку
                        if party_id not in self.models:
                            self.models[party_id] = {}
                        self.models[party_id][year] = model
                        
                        score = model.score(X_test, y_test)
                        print(f"    R² score: {score:.3f}")
    
    def predict_2026(self):
        # Берем последние известные данные
        latest_data = self.data[self.data['Year'] == 2021].iloc[0]
        
        # Простая экстраполяция для 2026 года
        prediction_data = latest_data.copy()
        for feature in self.features:
            if feature in ['GDP', 'Average_salary']:
                # Предполагаем рост 5% в год
                prediction_data[feature] *= (1.05 ** 5)
            elif feature in ['Population', 'Population_urban', 'Population_rural']:
                # Предполагаем небольшой рост 0.1% в год
                prediction_data[feature] *= (1.001 ** 5)
        
        # Делаем прогноз для каждой партии
        predictions = {}
        latest_parties = self.party_names[self.party_names['Year'] == 2021]
        
        for _, party_info in latest_parties.iterrows():
            party_id = party_info['Party_N']
            party_name = party_info['Party_name']
            
            if party_id in self.models and 2021 in self.models[party_id]:
                model = self.models[party_id][2021]  # Используем модель 2021 года
                X_pred = pd.DataFrame([prediction_data[self.features]])
                pred = model.predict(X_pred)[0]
                predictions[party_name] = max(0, min(100, pred))
        
        return predictions
    
    def visualize_results(self):
        # Получаем прогноз на 2026 год
        predictions_2026 = self.predict_2026()
        
        # Создаем график
        plt.figure(figsize=(15, 10))
        
        # Сортируем партии по убыванию процента голосов
        sorted_parties = sorted(predictions_2026.items(), key=lambda x: x[1], reverse=True)
        parties, percentages = zip(*sorted_parties)
        
        # Строим столбчатую диаграмму
        bars = plt.bar(parties, percentages)
        
        # Настраиваем внешний вид
        plt.title('Прогноз результатов выборов 2026 года', fontsize=14)
        plt.xlabel('Партии')
        plt.ylabel('Процент голосов')
        plt.xticks(rotation=45, ha='right')
        
        # Добавляем подписи значений над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Выводим важность признаков для основных партий
        self.plot_feature_importance()
    
    def plot_feature_importance(self):
        plt.figure(figsize=(12, 6))
        
        # Берем важность признаков для Единой России (party_id = 1) за 2021 год
        if 1 in self.models and 2021 in self.models[1]:
            importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.models[1][2021].feature_importances_
            })
            importance = importance.sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=importance)
            plt.title('Важность признаков для прогнозирования\n(на примере "Единой России")')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    predictor = ElectionPredictor() 