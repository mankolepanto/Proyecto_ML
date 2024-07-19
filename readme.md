# Detección de Billeteras Fraudulentas en la Red Ethereum

## Objetivo

Desarrollar un sistema para identificar y monitorizar billeteras fraudulentas en la red Ethereum, mejorando la seguridad y confianza en las transacciones de criptomonedas.

## Descripción del Proyecto

### 1. Recopilación de Datos
- **Fuente de Datos:** Utilización de APIs públicas y nodos de Ethereum para recopilar datos transaccionales.
- [transaction_dataset.csv](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset/data)
- **Alcance de los Datos:** Extracción de información sobre transacciones, historial de actividades de las billeteras y patrones de comportamiento.
#### FEATURES:

## Campos del Dataset

* **Index:** el número de índice de una fila  
* **Address:** la dirección de la cuenta de Ethereum  
* **FLAG:** si la transacción es fraudulenta o no  
* **Avg min between sent tnx:** tiempo promedio entre transacciones enviadas para la cuenta en minutos  
* **Avg min between received tnx:** tiempo promedio entre transacciones recibidas para la cuenta en minutos  
* **Time Diff between first and last (Mins):** diferencia de tiempo entre la primera y la última transacción  
* **Sent_tnx:** número total de transacciones normales enviadas  
* **Received_tnx:** número total de transacciones normales recibidas  
* **NumberofCreated_Contracts:** número total de transacciones de creación de contratos  
* **UniqueReceivedFrom_Addresses:** número total de direcciones únicas desde las cuales la cuenta recibió transacciones  
* **UniqueSentTo_Addresses20:** número total de direcciones únicas a las que la cuenta envió transacciones  
* **MinValueReceived:** valor mínimo en Ether recibido  
* **MaxValueReceived:** valor máximo en Ether recibido  
* **AvgValueReceived:** valor promedio en Ether recibido  
* **MinValSent:** valor mínimo en Ether enviado  
* **MaxValSent:** valor máximo en Ether enviado  
* **AvgValSent:** valor promedio en Ether enviado  
* **MinValueSentToContract:** valor mínimo en Ether enviado a un contrato  
* **MaxValueSentToContract:** valor máximo en Ether enviado a un contrato  
* **AvgValueSentToContract:** valor promedio en Ether enviado a contratos  
* **TotalTransactions (Including Tnxs to Create Contract):** número total de transacciones  
* **TotalEtherSent:** total de Ether enviado por la dirección de la cuenta  
* **TotalEtherReceived:** total de Ether recibido por la dirección de la cuenta  
* **TotalEtherSent_Contracts:** total de Ether enviado a direcciones de contrato  
* **TotalEtherBalance:** saldo total de Ether después de las transacciones realizadas  
* **TotalERC20Tnxs:** número total de transacciones de transferencia de tokens ERC20  
* **ERC20TotalEther_Received:** total de transacciones de tokens ERC20 recibidos en Ether  
* **ERC20TotalEther_Sent:** total de transacciones de tokens ERC20 enviados en Ether  
* **ERC20TotalEtherSentContract:** total de transferencias de tokens ERC20 a otros contratos en Ether  
* **ERC20UniqSent_Addr:** número de transacciones de tokens ERC20 enviadas a direcciones de cuenta únicas  
* **ERC20UniqRec_Addr:** número de transacciones de tokens ERC20 recibidas de direcciones únicas  
* **ERC20UniqRecContractAddr:** número de transacciones de tokens ERC20 recibidas de direcciones de contrato únicas  
* **ERC20AvgTimeBetweenSent_Tnx:** tiempo promedio entre transacciones de tokens ERC20 enviadas en minutos  
* **ERC20AvgTimeBetweenRec_Tnx:** tiempo promedio entre transacciones de tokens ERC20 recibidas en minutos  
* **ERC20AvgTimeBetweenContract_Tnx:** tiempo promedio entre transacciones de tokens ERC20 enviadas  
* **ERC20MinVal_Rec:** valor mínimo en Ether recibido de transacciones de tokens ERC20 para la cuenta  
* **ERC20MaxVal_Rec:** valor máximo en Ether recibido de transacciones de tokens ERC20 para la cuenta  
* **ERC20AvgVal_Rec:** valor promedio en Ether recibido de transacciones de tokens ERC20 para la cuenta  
* **ERC20MinVal_Sent:** valor mínimo en Ether enviado de transacciones de tokens ERC20 para la cuenta  
* **ERC20MaxVal_Sent:** valor máximo en Ether enviado de transacciones de tokens ERC20 para la cuenta  
* **ERC20AvgVal_Sent:** valor promedio en Ether enviado de transacciones de tokens ERC20 para la cuenta  
* **ERC20UniqSentTokenName:** número de tokens ERC20 únicos transferidos  
* **ERC20UniqRecTokenName:** número de tokens ERC20 únicos recibidos  
* **ERC20MostSentTokenType:** token más enviado para la cuenta a través de transacciones ERC20  
* **ERC20MostRecTokenType:** token más recibido para la cuenta a través de transacciones ERC20  

### 2. Análisis de Datos
- **Identificación de Patrones:** Uso de técnicas de minería de datos y análisis estadístico para identificar comportamientos atípicos y patrones asociados con actividades fraudulentas.
- **Características Claves:** Análisis de características como la frecuencia de las transacciones, montos transferidos, interacciones con billeteras conocidas y actividades sospechosas.

### 3. Desarrollo de Modelos Predictivos
- **Machine Learning:** Implementación de algoritmos de aprendizaje automático (como Random Forest, SVM, redes neuronales) para clasificar las billeteras como fraudulentas o legítimas.
- **Entrenamiento del Modelo:** Utilización de un conjunto de datos etiquetado para entrenar y validar los modelos predictivos.

### 4. Evaluación del Modelo
- **Métricas de Evaluación:** Medición de la precisión, sensibilidad, especificidad y área bajo la curva ROC para evaluar el rendimiento de los modelos.
- **Validación Cruzada:** Uso de técnicas de validación cruzada para asegurar la robustez y generalización del modelo.

### 5. Implementación del Sistema
- **Integración:** Desarrollo de una aplicación que integre el modelo predictivo para monitorear en tiempo real las transacciones en la red Ethereum.
- **Alertas y Reportes:** Generación de alertas automáticas y reportes cuando se detecten billeteras con actividades sospechosas.

### 6. Mejoras Continuas
- **Actualización de Datos:** Implementación de mecanismos para actualizar continuamente los datos de entrenamiento y el modelo con nueva información.
- **Feedback Loop:** Incorporación de feedback de los usuarios y análisis de casos reales para mejorar la precisión y reducir falsos positivos/negativos.

## Beneficios Esperados
- Reducción de fraudes y actividades ilícitas en la red Ethereum.
- Mejora en la confianza de los usuarios al realizar transacciones de criptomonedas.
- Contribución a la seguridad y estabilidad del ecosistema blockchain.

## Desafíos
- Manejo y procesamiento de grandes volúmenes de datos.
- Adaptación a las estrategias cambiantes de los actores fraudulentos.
- Garantía de baja tasa de falsos positivos para evitar perjuicios a usuarios legítimos.

