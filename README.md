<h1 align="center">Previsão de Tsunami após Terremoto com Algoritmos de Aprendizado de Máquina</h1>

## Resumo

Este repositório contém o trabalho final da disciplina Machine Learning do segundo semestre do curso de Bacharelado Interdisciplinar em Ciência e Tecnologia da Ilum - Escola de Ciência. O projeto consistiu em fazer 5 modelos preditivos otimizados via optuna para prever ocorrência de tsunamis após o evento de um terremoto. 

## Navegando pelo repositório

A pasta inicial apresenta o arquivo .ipynb 'Resultados_Discussão' é um notebook jupyter onde os modelos treinados são comparados. A pasta 'Dados' contém os dados brutos, um arquivo .ipynb chamado 'Pré-tratamento_dados' onde foi feito o pré-tratamento dos dados e contém uma outra pasta chamada 'dados_tratados' onde estão os DataFrames resultantes do pré-tratamento dos dados que são usados para treinar e testar os modelos. A pasta 'Modelos' contém um notebook jupyter para cada modelo, onde são otimizados, e um arquivo do tipo estudo do optuna para cada modelo, gerado para guardar informações sobre a otimização.

## Introdução ao projeto

O Banco de Dados Históricos Global de Tsunamis organizado pelo NOAA (*National Oceanic and Atmospheric and Administration*) dos EUA, apresenta que de 1850 até 2023 cerca de 89% dos tsunamis registrados foram causados por terremotos ou deslizamentos de terra causados por terremotos [1]. No entanto, não é qualquer terremoto que tem potencial de causar um tsunami. De forma geral, há quatro características comuns de terremotos que causaram tsunamis [2 e 3]:

- O terremoto deve ocorrer abaixo do oceano (profundidade raza - menos de 70 km abaixo da superfície) ou causar deslizamento de material no oceano;
- O terremoto deve ser forte, com pelo menos 6.5 de magnitude na escala Richter;
- O terremoto deve causar ruptura na superfície terrestre;
- O terremoto deve causar movimento vertical (até vários metros) no leito oceânico em uma grande área (até centenas de milhares de metros quadrados).

Tendo em vista a forte correlação entre a ocorrência desses dois eventos sísmicos e a causalidade previsível a depender das situações, este projeto final da disciplina de Machine Learning tem como objetivo estudar a aplicação de diferentes algoritmos de aprendizado de máquina para previsão de ocorrência de tsunami dado um terremoto. 

### 1. Sobre o *dataset* usado

Foi utilizado como fonte de dados o "*Global Earthquake-Tsunami Risk Assessment Dataset*" publicado no site *kaggle* [4]. As informações daqui foram - em sua maioria - retiradas da página do *dataset*.

#### 1.a Informações gerais

 - `Total de registros`: 782 terremotos
 - `Período de tempo`: 1º de janeiro de 2001 até 31 de dezembro de 2022 (22 anos)
 - `Cobertura geográfica`: Global (Latitude: -61.85° até 71.63°, Longitude: -179.97° até 179.66°)
 - `Formato do arquivo`: CSV
 - `Tamanho do arquivo`: ~41KB
 - `Valores faltantes`: None (100% complete dataset)
 - `Variável target`: Indicador de tsunami (classificação binária)

#### 1.b Classificação de ocorrência de Tsunami

 - `Eventos sem Tsunami`: 478 registros (61.1%)
 - `Eventos com Tsunami`: 304 registros (38.9%)

#### 1.c Distribuição de magnitude sísmica

 - `Intervalo`: 6.5 - 9.1 em escala Richter
 - `Magnitude média`: 6.94
 - `Maiores terremotos (≥8.0)`: 28 eventos, incluindo os mega-terremotos de 2004 (9.1) e de 2011 (9.1)

#### 1.d Sobre as colunas

 - `magnitude`: float. Indica a magnitude do terremoto na escala Richter, portanto, a energia liberada por ele. Varia entre 6.5 e 9.1.
 - `cdi`: int. Indica a intensidade sentida pela comunidade atingida (do inglês, *Community Internet Intensity*). Este valor é obtido pela USGS (do inglês *United States Geological Survey*, ou Serviço Geológico dos Estados Unidos em português) através de questionários realizados com a população atingida e visa servir de comparação à escala de magnitude, completando os dados com informação sobre como o terremoto foi percebido. Essa escala varia entre 1 e 10. Como alguns terremotos possuem cdi 0 no dataset, é preciso retirá-los, pois indicam NaN. [5 e 6]
 - `mmi`: int. Chamado *Modified Mercalli Intensity* em inglês, ou Intensidade de Mercalli Modificada em português. Mede os efeitos de um terremoto, ou seja, descreve como o tremor impactou a população e a infraestrutura, dependendo de fatores como composição do solo local e não apenas de valores objetivos específicos ao terremoto (como escala Richter). Essa escala varia entre 1 e 9. [7]
 - `sig`: int. Um número que descreve o quão significante foi o evento. Quanto maior, mais significante. O valor foi determinado com base em vários fatores, incluindo: magnitude, máximo MMI, reportes, e impacto estimado. Esse valor vai de 650 até 2910.
 - `nst`: int. Um número que indica a quantidade total de estações de monitoramento sísmicos usadas para determinar a localização do terremoto. Esse valor vai de 0 até 934.
 - `dmin`: float. A distância (em graus - ângulo) do epicentro até a estação de monitoramento sísmico mais próxima. Esse valor vai de 0 até 17.7.
 - `gap`: float. A maior diferença angular azimutal entre estações azimutalmente adjacentes (em graus - ângulo). De forma geral, mede o quão bem coberta por estações de monitoramento sísmico é a região ao redor do epicentro. Quanto maior é essa distância angular chamada 'gap', pior a cobertura e mais incertezas a respeito da localização do epicentro. Esse valor vai de 0 até 239.
 - `depth`: float. A profundidade do epicentro (em km). Esse valor vai de 2.7 até 670.8.
 - `latitude`: float. Latitude da coordenada do epicentro do terremoto. Esse valor vai de -61.85 até 71.63.
 - `longitude`: float. Longitude da coordenada do epicentro do terremoto. Esse valor vai de -179.97 até 179.66.
 - `Year`: int. Ano de ocorrência do terremoto. Esse valor vai de 2001 até 2022.
 - `Month`: int. Mês de ocorrência do terremoto. Esse valor vai de 1 até 12.
 - `tsunami`: int (binário). Indica ocorrência de tsunami (1) ou não (0).

### 2. Sobre os algoritmos de aprendizado de máquina usados

Foram usados 5 algoritmos de aprendizado de máquina (mais o Baseline): kNN classificador, congresso de kNNs, regressão logística, floresta aleatória classificadora e SVC. Esses foram os modelos escolhidos, pois foram estudados durante a disciplina. Para saber mais sobre eles, a documentação do scikit-learn para cada um está nas referências. Exceto pelo congresso de kNNs, há um artigo sobre. [8-12]

### 3. Sobre a métrica de desempenho para otimização

Para "guiar" o modelo na otimização pela validação cruzada interna do optuna, foi escolhida a **métrica F$_\beta$**. Ela permite fazer um balanço entre Recall e Precision. Considerando o conjunto de dados estudado e o objetivo de prever a ocorrência de tsunamis, tem-se um dataset desbalanceado em que Falsos Negativos (FN) são mais custosos (são prioridade) que Falsos Positivos (FP). 

Similarmente a diagnósticos médicos, prever que não haverá tsunami, não soar o alarme e ele acontecer (FN) é muito mais catastrófico do que prever que haverá tsunami, soar o alarme, evacuar a cidade e ele não acontecer (FP). No entanto, ao mesmo tempo que se deve penalizar FN mais do que FP, não se pode abrir mão destes, pois uma evacuação é também custosa. 

Dessa forma, a métrica escolhida reflete esse desbalanço, pois a F$_\beta$ que faz o cálculo:

$$
F_{\beta} = \frac{(1 + \beta^2)\, \mathrm{VP}}{(1 + \beta^2)\, \mathrm{VP} + \mathrm{FP} + \beta^2\, \mathrm{FN}}
$$

Onde
- TP: Verdadeiros Positivos;
- FP: Falsos Positivos;
- FN: Falsos Negativos.

Note que $\beta$ age como um peso para FN, por isso aumentar seu valor para além de 1 indica priorizar FN. Não foi feito um estudo sobre o valor mais adequado de beta para este caso, portanto, foi escolhido o valor recomendado pela documentação do scikit-learn quando se deseja priorizar FN: $\beta$ = 2. [13]

## Resultados e Discussão

## Conclusão

## Referências teóricas:

[1] *Tsunami Generation: Earthquakes*. Site do NOAA (*National Oceanic and Atmospheric Administration*). Acesso em: 01/11/2025. Disponível em: https://www.noaa.gov/jetstream/tsunamis/tsunami-generation-earthquakes

[2] *Do all earthquakes cause tsunamis?*. Site da UWI-SRC (*University of the West Indies Seismic Research Centre*). Acesso em: 01/11/2025. Disponível em: https://uwiseismic.com/sp_faq/do-all-earthquakes-cause-tsunamis/

[3] *What causes Tsunamis*. Site da Unesco (Organização das Nações Unidas para a Educação, a Ciência e a Cultura). Acesso em: 01/11/2025. Disponível em: https://legacy2.ctic.ioc-unesco.org/tsunami-info/what-causes-tsunamis

[4] *Global Earthquake-Tsunami Risk Assessment Dataset*. Página no site *kaggle*. Acesso em: 01/11/2025. Disponível em: https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset 

[5] *About Community Internet Intensity Maps*. Site do governo do Canadá. Acesso em: 01/11/2025. Disponível em: https://www.earthquakescanada.nrcan.gc.ca/dyfi-lavr/about-en.php#ciim

[6] Wald, D.J., Wald, L., Dewey, J.W., Quitoriano, V., & Adams, E. (2001). *Did you feel it? : Community-made earthquake shaking maps* (Fact Sheet 030-01). U.S. Geological Survey. DOI:10.3133/fs03001

[7] *The Modified Mercalli Intensity Scale*. Site da USGS (*United States Geological Survey*). Acesso em: 01/11/2025. Disponível em: https://www.usgs.gov/programs/earthquake-hazards/modified-mercalli-intensity-scale?utm_source=chatgpt.com 

[8] **kNN classificador**. Documentação do scikit-learn. Acesso em: 01/11/2025. Disponível em: https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification

[9] **Congresso de kNNs**. Bao, Y., Ishii, N., & Du, X. (2004). Combining Multiple k-Nearest Neighbor Classifiers Using Different Distance Functions. In Z. R. Yang et al. (Eds.), IDEAL 2004: Intelligent Data Engineering and Automated Learning (Lecture Notes in Computer Science, Vol. 3177, pp. 634–641). Springer-Verlag Berlin Heidelberg.
https://doi.org/10.1007/978-3-540-28651-6_93

[10] **Regressão Logística**. Documentação do scikit-learn. Acesso em: 01/11/2025. Disponível em: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

[11] **Floresta Aleatória Classificadora**. Documentação do scikit-learn. Acesso em: 01/11/2025. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

[12] **SVC**. Documentação do scikit-learn. Acesso em: 01/11/2025. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

[13] Página sobre F-score da documentação do scikit-learn. Acesso em: 31/10/2025. Disponível em: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures 
