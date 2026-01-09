import json
import math
import os
import sqlite3
from datetime import datetime
import requests

PREDICTION_PATH = "https://ml-stock-market-predictor.onrender.com/predict"
PREDICTION_HEADERS = {"Content-Type": "application/json"}
PREDICTION_BODY={"symbol":"PETR4.SA"}


class AgenteConselheiroDeAcoes:
    """
    Agente responsável por analisar histórico de preços e recomendar ações financeiras.

    Esse agente é puramente estatístico e matemático, não se baseando em fatores externos.
    Ele utiliza um modelo LSTM para previsões e ajusta sua política de operação
    dinamicamente com base nos erros e acertos passados (Aprendizado Adaptativo).

    Attributes:
        policy_file (str): Caminho para o arquivo de política do agente.
        metrics_file (str): Caminho para o arquivo de métricas do agente.
        db_file (str): Caminho para o arquivo do banco de dados SQLite do agente.
        policy (dict[str, float]): Política carregada do arquivo de configuração.
    """

    def __init__(self) -> None:
        """
        Inicializa o agente com arquivos de configuração e o banco de dados.

        Este método carrega os arquivos de configuração (política e métricas) e
        inicializa o banco de dados do agente. Caso os arquivos de configuração
        não existam, são criados com valores padrão.
        """
        self.policy_file = "policy.json"
        self.metrics_file = "metrics.json"
        self.db_file = "memory.db"
        self.policy = self._load_policy()
        self._init_db()

    def _load_policy(self) -> dict[str, float]:
        """
        Carrega a política do agente a partir do arquivo "policy.json" (se existir).

        O "policy.json" contém os parâmetros necessários para o correto funcionamento do agente.
        Caso o arquivo não exista ainda, será criado um com os valores padrão:
        - threshold: 0.01 (Limite de variação de preço para realizar uma ação)
        - learning_rate: 0.05 (Taxa de aprendizado usada para ajustar o limiar de ação)
        - min_threshold: 0.002 (Valor mínimo do limiar de ação)
        - max_threshold: 0.05 (Valor máximo do limiar de ação)

        Returns
        -------
        dict
            [str, float]

        Where:
        - str: Nome da configuração
        - float: Valor da configuração
        """
        if os.path.exists(self.policy_file):
            try:
                with open(self.policy_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"AVISO: O arquivo '{self.policy_file}' está corrompido ou vazio.")
                print("Corrija-o e tente novamente ou apague-o para usar a política padrão.")
                raise
            except Exception as e:
                print(f"ERRO INESPERADO ao ler política: {e}")
                raise

        default_policy = {
            "threshold": 0.01,
            "learning_rate": 0.05,
            "min_threshold": 0.002,
            "max_threshold": 0.05,
        }
        self._save_policy(default_policy)
        return default_policy

    def _save_policy(self, policy_data: dict[str, float] | None = None) -> None:
        """
        Salva a política do agente no arquivo "policy.json".

        Se policy_data for fornecido, atualiza a política do agente com os dados fornecidos.
        Caso contrário, salva a política atual no arquivo.

        Parameters
        ----------
        policy_data : dict[str, float] | None
            Dados opcionais para sobrescrever a política atual.
            A estrutura deve ser { "nome_config" (str): valor (float) }.
            Ex: {"threshold": 0.02}.
            O padrão é None (apenas salva o estado atual).
        """
        if policy_data:
            self.policy = policy_data
        try:
            with open(self.policy_file, "w") as f:
                json.dump(self.policy, f, indent=4)
        except Exception as e:
            print(f"ERRO INESPERADO ao salvar política: {e}")
            raise

    def _init_db(self) -> None:
        """
        Inicializa o banco de dados "memory.db" com a tabela "history".

        Cria a tabela "history", se ela não existir ainda, com as seguintes colunas:
        - id: Identificador único da decisão (chave primária)
        - date: Data da decisão (formato "YYYY-MM-DD HH:MM:SS")
        - price_at_decision: Preço da ação no momento da decisão
        - predicted_price: Preço previsto pelo modelo
        - action: Ação tomada pelo agente (COMPRAR, VENDER ou MANTER)
        - threshold_used: Limiar de segurança utilizado na decisão
        - volatility: Variação de preço no momento da decisão
        - learning_rate_used: Taxa de aprendizado no momento usada para ajustar o limiar do agente (threshold)
        - real_return: Retorno real da operação dada a decisão (ganho, perda ou nulo)
        - outcome_evaluated: Indica se a decisão foi avaliada ou não (1 para sim, 0 para não)

        Se a tabela "history" já existir, completa o banco de dados.
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                price_at_decision REAL,
                predicted_price REAL,
                action TEXT,
                threshold_used REAL,
                volatility REAL, 
                learning_rate_used REAL,
                real_return REAL,
                outcome_evaluated INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def _record_decision(self, price: float, pred: float, action: str, vol: float) -> None:
        """
        Registra uma decisão tomada pelo agente no banco de dados "memory.db".

        Parameters
        ----------
        price : float
            Preço da ação no momento da decisão
        pred : float
            Preço previsto pelo modelo
        action : str
            Ação tomada pelo agente (COMPRAR, VENDER ou MANTER)
        vol : float
            Variação de preço no momento da decisão
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO history (date, price_at_decision, predicted_price, action, threshold_used, volatility, learning_rate_used, real_return, outcome_evaluated)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 0)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                price,
                pred,
                action,
                self.policy["threshold"],
                vol,
            ),
        )
        conn.commit()
        conn.close()

    def _calcular_volatilidade(self, prices: list, window: int = 5) -> float:
        """
        Calcula a volatilidade dos últimos cinco dias da lista de preços históricos do mercado de ações.

        Parameters
        ----------
        prices : list
            Lista de preços históricos
        window : int, default=5
            Janela de observação para calcular a volatilidade (default=5, 5 dias, segunda a sexta, observação semanal)

        Returns
        -------
        float
            Volatilidade do intervalo dos ultimos cinco dias do mercado de ações
        """
        if len(prices) < window:
            return 0.01  # Valor padrão se não tiver dados

        subset = prices[-window:]
        avg = sum(subset) / len(subset)
        variance = sum((x - avg) ** 2 for x in subset) / (len(subset) - 1)
        std_dev = math.sqrt(variance)

        return std_dev / avg

    def _atualizar_metricas(self) -> None:
        """
        Atualiza as métricas do agente, como a taxa de acerto (win rate) e o lucro acumulado estimado.

        As métricas são salvas em um arquivo JSON chamado "metrics.json", com as seguintes chaves:
        - ultima_atualizacao: Data e hora da última atualização nesse arquivo
        - total_operacoes_avaliadas: Número total de operações avaliadas (compras, vendas e manter)
        - trades_ativos: Número de trades ativos (compras e vendas)
        - taxa_de_acerto: Taxa de acerto (win rate) do agente, em porcentagem
        - lucro_acumulado_estimado: Lucro acumulado estimado do agente, em porcentagem
        - threshold_atual: Nível de exigência (threshold) atual do agente
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Pegando apenas trades já finalizados (avaliados) e que não foram "MANTER"
        # (Consideramos que "MANTER" é neutro, lucro = 0)
        cursor.execute("SELECT action, real_return FROM history WHERE outcome_evaluated = 1")
        trades = cursor.fetchall()
        conn.close()

        total_trades = len(trades)
        if total_trades == 0:
            return  # Sem dados ainda

        acertos = 0
        lucro_acumulado_pct = 0.0

        for action, ret in trades:
            if action == "MANTER":
                continue  # Ignorando trades com "MANTER" do Win Rate

            # Se o retorno foi positivo, conta como acerto (podendo ser compra ou venda)
            if ret > 0:
                acertos += 1

            # Soma simples de porcentagem
            lucro_acumulado_pct += ret

        # Win Rate (Taxa de Acerto)
        # Filtra apenas compras e vendas para o Win Rate
        trades_ativos = [t for t in trades if t[0] != "MANTER"]
        win_rate = (acertos / len(trades_ativos)) if len(trades_ativos) > 0 else 0.0

        metrics = {
            "ultima_atualizacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_operacoes_avaliadas": total_trades,  # Compras, Vendas e Manter
            "trades_ativos": len(trades_ativos),  # Compras e Vendas apenas
            "taxa_de_acerto": round(win_rate * 100, 2),  # Taxa de acerto dos trades ativos (%)
            "lucro_acumulado_estimado": round(lucro_acumulado_pct * 100, 2),  # (%)
            "threshold_atual": self.policy["threshold"],
        }

        try:
            with open(self.metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(f"ERRO INESPERADO ao salvar métricas: {e}")
            print("O arquivo 'metrics.json' não foi atualizado!")

    def learn(self, current_price_today: float) -> tuple[str, float] | tuple[None, float]:
        """
        Aprende com a última decisão tomada pelo agente.

        Se a última decisão foi COMPRAR ou VENDER e a variação real foi negativa, aumenta o rigor da política.
        Se a última decisão foi MANTER e a variação real foi maior que o rigor da política, diminui o rigor da política.

        Parameters
        ----------
        current_price_today : float
            Preço atual da ação

        Returns
        -------
        str
            Mensagem com o resultado do aprendizado
        float
            Variação real entre o preço de ontem e o preço de hoje
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, action, price_at_decision FROM history WHERE outcome_evaluated = 0 ORDER BY id DESC LIMIT 1"
        )
        last_trade = cursor.fetchone()

        if not last_trade:
            conn.close()
            return None, 0

        trade_id, action, price_yesterday = last_trade
        variacao_real = (current_price_today - price_yesterday) / price_yesterday

        # Aposta do agente. Comprar achando que vai subir, vender achando que vai cair, manter se nenhuma das opções for verdadeira
        lucro_trade = 0.0
        if action == "COMPRAR":
            lucro_trade = variacao_real  # Ganha na subida (ou perde se cair)
        elif action == "VENDER":
            lucro_trade = -variacao_real  # Ganha na queda (ou perde se subir)
        else:  # MANTER, não ganha nada (pode ter custo oportunidade)
            lucro_trade = 0.0

        # Se a variação foi grande, aprendemos mais rápido. Se foi pequena, ajuste fino.
        base_lr = self.policy["learning_rate"]
        dynamic_lr = base_lr * (1 + abs(variacao_real) * 10)
        # Trava para o LR não explodir (fazendo com que não aprenda muito rápido)
        dynamic_lr = min(dynamic_lr, 0.20)

        msg = "Mantendo rigor. Nenhuma oportunidade encontrada."  # Comportamento padrão
        adjusted = False  # Flag para ajuste de threshold

        # Lógica de Aprendizado
        if (action == "COMPRAR" and variacao_real < 0) or (
            action == "VENDER" and variacao_real > 0
        ):
            self.policy["threshold"] *= 1 + dynamic_lr
            msg = f"Aumentou rigor (Erro  de recomendação). Dynamic_lr usado: {dynamic_lr:.4f}"
            adjusted = True

        elif action == "MANTER" and abs(variacao_real) > self.policy["threshold"]:
            self.policy["threshold"] *= 1 - dynamic_lr
            msg = f"Diminuiu rigor (Perdeu Oportunidade). Dynamic_lr usado: {dynamic_lr:.4f}"
            adjusted = True

        # Faixa de operação do threshold (0.02 a 5%)
        if adjusted:
            self.policy["threshold"] = max(
                self.policy["min_threshold"],
                min(self.policy["max_threshold"], self.policy["threshold"]),
            )

        self._save_policy()

        cursor.execute(
            "UPDATE history SET outcome_evaluated = 1, learning_rate_used = ?, real_return = ? WHERE id = ?",
            (dynamic_lr, lucro_trade, trade_id),
        )
        conn.commit()
        conn.close()

        self._atualizar_metricas()

        return msg, variacao_real

    def decide(self, market_history: list) -> tuple[str, float, float]:
        """
        Decide a recomendação (comprar, vender ou manter) com base nos dados do mercado.

        Parameters
        ----------
        market_history : list
            Lista de preços históricos do mercado de ações

        Returns
        -------
        tuple
            (action, predicted_price, delta)

        Where:
        - action (str): Ação tomada (comprar, vender ou manter)
        - predicted_price (float): Preço previsto pelo modelo
        - delta (float): Variação entre o preço previsto e o preço atual
        """
        current_price = market_history[-1]

        # Lendo predição de saída gerada pela LSTM
        try:
            prediction_response = requests.post(
                PREDICTION_PATH,
                headers=PREDICTION_HEADERS,
                json=PREDICTION_BODY,
                timeout=60,
                max_retries=10,
            )
            predicted_price = prediction_response.json().get("value")
        except Exception as e:
            print(f"ERRO ao ler predição: {e}")
            raise

        # Calcula volatilidade dos últimos cinco dias (semanal)
        volatility = self._calcular_volatilidade(market_history)
        delta = (predicted_price - current_price) / current_price

        # O Threshold efetivo é a política base + a sensibilidade da volatilidade
        # Se o mercado varia 2% ao dia, exigir 1% é pouco. O agente se adapta.
        threshold_base = self.policy["threshold"]
        threshold_efetivo = max(threshold_base, volatility * 0.5)

        action = "MANTER"
        if delta > threshold_efetivo:
            action = "COMPRAR"
        elif delta < -threshold_efetivo:
            action = "VENDER"

        # Registra a decisão tomada pelo agente
        self._record_decision(current_price, predicted_price, action, volatility)

        return action, predicted_price, delta
