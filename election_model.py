import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.features = ['Year', 'Population', 'Population_urban', 'Population_rural', 
                        'Poverty', 'Unemployment', 'Employment', 'GDP', 
                        'Inflation', 'Average_salary']
        
        # Заполняем пропущенные значения средними по году
        for feature in self.features[1:]:  # Пропускаем Year
            mean_values = self.election_data.groupby('Year')[feature].transform('mean')
            self.election_data[feature] = self.election_data[feature].fillna(mean_values)
        
        # Добавляем производные признаки
        self.election_data['Urban_ratio'] = self.election_data['Population_urban'] / self.election_data['Population']
        self.election_data['GDP_per_capita'] = self.election_data['GDP'] / self.election_data['Population']
        self.election_data['Poverty_rate'] = self.election_data['Poverty'] / self.election_data['Population']
        
        # Обновляем список признаков
        self.features.extend(['Urban_ratio', 'GDP_per_capita', 'Poverty_rate'])
        
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
        
        # Нормализация данных
        self.scaler = StandardScaler()
        
        print("Обучение моделей:")
        # Получаем список всех партий из 2021 года
        latest_parties = self.party_names[self.party_names['Year'] == 2021]
        
        for _, party_info in latest_parties.iterrows():
            party_id = party_info['Party_N']
            party_name = party_info['Party_name']
            party_col = f'Party_{party_id}_percent'
            
            if party_col in self.election_data.columns:
                # Берем все данные для партии
                X = self.election_data[self.features]
                y = self.election_data[party_col]
                
                # Убираем строки с пропущенными значениями
                mask = y.notna()
                X = X[mask]
                y = y[mask]
                
                if len(y) > 0:
                    print(f"Партия {party_name}: {len(y)} наблюдений")
                    
                    # Нормализация признаков (кроме года)
                    X_scaled = X.copy()
                    X_scaled[self.features[1:]] = self.scaler.fit_transform(X[self.features[1:]])
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42
                    )
                    
                    # Используем GradientBoostingRegressor с оптимизированными параметрами
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=4,
                        min_samples_split=10,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    
                    self.models[party_id] = {
                        'model': model,
                        'scaler': self.scaler,
                        'features': self.features
                    }
                    
                    score = model.score(X_test, y_test)
                    print(f"    R² score: {score:.3f}")
    
    def predict_2026(self, region='Вся Россия'):
        """Прогнозирует результаты для 2026 года для конкретного региона"""
        # Берем последние известные данные для выбранного региона
        if region == 'Вся Россия':
            latest_data = self.data[self.data['Year'] == 2021].iloc[0].copy()
        else:
            latest_data = self.data[
                (self.data['Year'] == 2021) & 
                (self.data['Region'] == region)
            ].iloc[0].copy()
        
        prediction_data = latest_data.copy()
        prediction_data['Year'] = 2026
        
        # Экстраполируем значения с учетом региональной специфики
        prediction_data['GDP'] *= (1.05 ** 5)  # Рост ВВП на 5% в год
        prediction_data['Average_salary'] *= (1.05 ** 5)  # Рост зарплат на 5% в год
        prediction_data['Population'] *= (1.001 ** 5)  # Небольшой рост населения
        prediction_data['Population_urban'] *= (1.001 ** 5)
        prediction_data['Population_rural'] *= (1.001 ** 5)
        
        # Рассчитываем производные признаки
        prediction_data['Urban_ratio'] = prediction_data['Population_urban'] / prediction_data['Population']
        prediction_data['GDP_per_capita'] = prediction_data['GDP'] / prediction_data['Population']
        prediction_data['Poverty_rate'] = prediction_data['Poverty'] / prediction_data['Population']
        
        predictions = {}
        latest_parties = self.party_names[self.party_names['Year'] == 2021]
        
        for _, party_info in latest_parties.iterrows():
            party_id = party_info['Party_N']
            party_name = party_info['Party_name']
            
            if party_id in self.models:
                model_data = self.models[party_id]
                X_pred = pd.DataFrame([prediction_data[model_data['features']]])
                
                # Нормализуем признаки (кроме года)
                X_pred_scaled = X_pred.copy()
                X_pred_scaled[model_data['features'][1:]] = model_data['scaler'].transform(X_pred[model_data['features'][1:]])
                
                pred = model_data['model'].predict(X_pred_scaled)[0]
                predictions[party_name] = max(0, min(100, pred))
        
        return predictions
    
    def predict_future(self, year, region='Вся Россия'):
        """Прогнозирует результаты для будущего года"""
        if year == 2026:
            return self.predict_2026(region)
        
        # Для других лет используем экстраполяцию от 2026
        predictions_2026 = self.predict_2026(region)
        years_diff = year - 2026
        
        # Простая линейная экстраполяция с учетом региональной специфики
        predictions = {}
        for party, value_2026 in predictions_2026.items():
            # Предполагаем небольшое изменение (±1% в год)
            change = np.random.uniform(-1, 1) * years_diff
            pred = value_2026 + change
            predictions[party] = max(0, min(100, pred))
        
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
        if 1 in self.models:
            importance = pd.DataFrame({
                'feature': self.models[1]['features'],
                'importance': self.models[1]['model'].feature_importances_
            })
            importance = importance.sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=importance)
            plt.title('Важность признаков для прогнозирования\n(на примере "Единой России")')
            plt.tight_layout()
            plt.show()

    def get_historical_result(self, party_name, year, region):
        """Получает исторический результат партии"""
        try:
            party_id = self.party_names[
                (self.party_names['Year'] == year) & 
                (self.party_names['Party_name'] == party_name)
            ]['Party_N'].iloc[0]
            
            data = self.data[
                (self.data['Year'] == year) & 
                (self.data['Region'] == region if region != 'Вся Россия' else True)
            ]
            
            if not data.empty:
                return data[f'Party_{party_id}_percent'].iloc[0]
        except:
            pass
        return None

class ElectionGUI:
    def __init__(self, predictor):
        self.predictor = predictor
        self.root = tk.Tk()
        self.root.title("Прогноз результатов выборов")
        self.root.geometry("1200x800")
        
        # Создаем фреймы
        self.create_control_frame()
        self.create_graph_frame()
        
        # Сразу показываем прогноз для всей России на 2026
        self.show_initial_forecast()
    
    def create_control_frame(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Выпадающий список для выбора года
        ttk.Label(control_frame, text="Год:").pack(side=tk.LEFT, padx=5)
        # Только года с выборами и прогнозные года
        years = [2011, 2016, 2021] + list(range(2026, 2041, 5))
        self.year_var = tk.StringVar(value="2026")
        year_cb = ttk.Combobox(control_frame, textvariable=self.year_var, values=years, width=10)
        year_cb.pack(side=tk.LEFT, padx=5)
        
        # Выпадающий список для выбора партии
        ttk.Label(control_frame, text="Партия:").pack(side=tk.LEFT, padx=5)
        parties = ['Все партии'] + sorted(set(self.predictor.party_names[
            self.predictor.party_names['Year'] == 2021]['Party_name']))
        self.party_var = tk.StringVar(value="Все партии")
        party_cb = ttk.Combobox(control_frame, textvariable=self.party_var, values=parties, width=30)
        party_cb.pack(side=tk.LEFT, padx=5)
        
        # Выпадающий список для выбора региона
        ttk.Label(control_frame, text="Регион:").pack(side=tk.LEFT, padx=5)
        regions = ['Вся Россия'] + sorted(self.predictor.data['Region'].unique())
        self.region_var = tk.StringVar(value="Вся Россия")
        region_cb = ttk.Combobox(control_frame, textvariable=self.region_var, values=regions, width=30)
        region_cb.pack(side=tk.LEFT, padx=5)
        
        # Кнопка построения прогноза
        ttk.Button(control_frame, text="Построить прогноз", 
                  command=self.update_forecast).pack(side=tk.LEFT, padx=5)
    
    def create_graph_frame(self):
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем область для графика
        self.fig = plt.Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_initial_forecast(self):
        """Показывает начальный прогноз для всей России на 2026 год"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Получаем прогноз
        predictions = self.predictor.predict_2026()
        
        # Сортируем партии по убыванию процента голосов
        sorted_parties = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        parties, percentages = zip(*sorted_parties)
        
        # Строим столбчатую диаграмму
        bars = ax.bar(parties, percentages)
        
        # Настраиваем внешний вид
        ax.set_title('Прогноз результатов выборов 2026 года\nВся Россия', fontsize=14)
        ax.set_xlabel('Партии')
        ax.set_ylabel('Процент голосов')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Добавляем подписи значений над столбцами
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_forecast(self):
        """Обновляет прогноз на основе выбранных параметров"""
        year = int(self.year_var.get())
        party = self.party_var.get()
        region = self.region_var.get()
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        try:
            if year in [2011, 2016, 2021]:  # Только для лет с выборами
                # Показываем реальные данные
                data = self.predictor.data[
                    (self.predictor.data['Year'] == year) & 
                    (self.predictor.data['Region'] == region if region != 'Вся Россия' else True)
                ]
                
                if len(data) == 0:
                    raise ValueError(f"Нет данных для {region} за {year} год")
                
                if party == 'Все партии':
                    # Собираем данные по всем партиям
                    party_results = {}
                    for i in range(1, 15):
                        col = f'Party_{i}_percent'
                        if col in data.columns and not pd.isna(data[col].iloc[0]):
                            party_name = self.predictor.party_names[
                                (self.predictor.party_names['Year'] == year) & 
                                (self.predictor.party_names['Party_N'] == i)
                            ]['Party_name'].iloc[0]
                            party_results[party_name] = data[col].iloc[0]
                    
                    if not party_results:
                        raise ValueError(f"Нет данных о результатах партий за {year} год")
                    
                    # Сортируем и отображаем
                    sorted_parties = sorted(party_results.items(), key=lambda x: x[1], reverse=True)
                    parties, percentages = zip(*sorted_parties)
                    
                    # Строим столбчатую диаграмму
                    bars = ax.bar(parties, percentages)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                    # Добавляем подписи значений
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom')
                else:
                    # Показываем динамику для выбранной партии
                    years = [2011, 2016, 2021]
                    percentages = []
                    
                    for y in years:
                        val = self.predictor.get_historical_result(party, y, region)
                        percentages.append(val if pd.notna(val) else None)
                    
                    # Убираем None значения
                    valid_data = [(y, p) for y, p in zip(years, percentages) if p is not None]
                    if valid_data:
                        plot_years, plot_percentages = zip(*valid_data)
                        ax.plot(plot_years, plot_percentages, 'bo-', label='Исторические данные')
                        ax.legend()
            
            elif year > 2024:  # Для прогнозных данных
                predictions = self.predictor.predict_future(year, region)
                
                if party == 'Все партии':
                    # Сортируем партии по убыванию процента голосов
                    sorted_parties = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    parties, percentages = zip(*sorted_parties)
                    
                    # Строим столбчатую диаграмму
                    bars = ax.bar(parties, percentages)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                    # Добавляем подписи значений
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom')
                else:
                    # Показываем динамику для выбранной партии
                    years = [2011, 2016, 2021]
                    percentages = []
                    
                    # Получаем исторические данные
                    for y in years:
                        val = self.predictor.get_historical_result(party, y, region)
                        percentages.append(val if pd.notna(val) else None)
                    
                    # Убираем None значения
                    valid_data = [(y, p) for y, p in zip(years, percentages) if p is not None]
                    if valid_data:
                        plot_years, plot_percentages = zip(*valid_data)
                        ax.plot(plot_years, plot_percentages, 'bo-', label='Исторические данные')
                    
                    # Добавляем прогноз
                    years.append(year)
                    percentages.append(predictions[party])
                    ax.plot([years[-2], years[-1]], [percentages[-2], percentages[-1]], 'r--', label='Прогноз')
                    ax.legend()
            else:
                raise ValueError("Для этого года нет данных о выборах")
            
            # Настраиваем заголовок и подписи осей
            title = f'{"Прогноз" if year > 2024 else "Результаты"} выборов {year} года'
            if region != 'Вся Россия':
                title += f'\n{region}'
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Год' if party != 'Все партии' else 'Партии')
            ax.set_ylabel('Процент голосов')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            tk.messagebox.showerror("Ошибка", f"Ошибка при построении прогноза: {str(e)}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    predictor = ElectionPredictor()
    gui = ElectionGUI(predictor)
    gui.run() 