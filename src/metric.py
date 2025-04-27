class FinEffect:
    def __init__(self, key_rate=0.2, n_days=365, delta_deriv=0.005, delta_cred_cb=0.01, delta_dep_cb=-0.009):
        """
        Подсчет финэффекта от решений модели
        
        Args:
            key_rate (float): Ключевая ставка ЦБ
            n_days (int): база для расчета дневной ставки
            delta_deriv (float): разница ставок с рынком деривативов
            delta_cred_cb (float): разница по overnight кредитованию 
            delta_dep_cb (float): разница по overnight депозитам 
        """
        self.key_rate = key_rate
        self.n_days = n_days
        self.delta_deriv = delta_deriv
        self.delta_cred_cb = delta_cred_cb
        self.delta_dep_cb = delta_dep_cb
    
    def model_value(self, predicted_balances, actual_balances):
        """
        Расчет эффекта от прогнозирования ликвидности
        
        Args:
            predicted_balances (np.array): Массив прогнозируемых сальдо по дням
            actual_balances (np.array): Массив фактических сальдо по дням

        Return:
            total_effect (float): Суммарный эффект за весь период
        """

        if len(predicted_balances) != len(actual_balances):
            raise ValueError("Длины массивов прогноза и факта должны совпадать")
        
        total_effect = 0.0
        
        for pred, actual in zip(predicted_balances, actual_balances):
            if pred > 0:
                decision_effect = pred * (self.key_rate + self.delta_deriv) / self.n_days
            else:
                decision_effect = pred * (self.key_rate + self.delta_deriv) / self.n_days
            
            if (actual >= pred) and (pred >=0):
                adjustment = (actual - pred) * (self.key_rate + self.delta_dep_cb) / self.n_days
            elif (actual <= pred) and (pred >= 0):
                adjustment = (actual - pred) * (self.key_rate + self.delta_cred_cb) / self.n_days
            elif (actual >= pred) and (pred <= 0):
                adjustment = (actual - pred) * (self.key_rate + self.delta_dep_cb) / self.n_days
            elif (actual <= pred) and (pred <= 0):
                adjustment = (actual - pred) * (self.key_rate + self.delta_cred_cb) / self.n_days
            else:
                adjustment = 0
            
            daily_effect = decision_effect + adjustment

            #print(daily_effect)
            total_effect += daily_effect
        
        return total_effect

    def base_value(self, actual_balances):
        """
        Расчет эффекта без прогнозирования ликвидности
        
        Args:
            actual_balances (np.array): Массив фактических сальдо по дням

        Return:
            total_effect (float): Суммарный эффект за весь период
        """
        total_effect = 0

        for actual in actual_balances:
            if actual > 0:
                daily_effect = actual * (self.key_rate + self.delta_dep_cb) / self.n_days
            else:
                daily_effect = actual * (self.key_rate + self.delta_cred_cb) / self.n_days
            
            #print(daily_effect)
            total_effect += daily_effect
        return total_effect
    
    def model_effct(self, predicted_balances, actual_balances):
        """
        Расчет чистого эффекта от прогнозирования ликвидности по отношению к поведению без прогнощирования
        
        Args:
            predicted_balances (np.array): Массив прогнозируемых сальдо по дням
            actual_balances (np.array): Массив фактических сальдо по дням

        Return:
            total_effect (float): Суммарный эффект за весь период
        """
        model_val = self.model_value(predicted_balances, actual_balances)
        base_val = self.base_value(actual_balances)

        effect = model_val - base_val
        return effect