# Relatório Técnico --- Reconhecimento de Comandos de Voz com Edge AI no ESP32-S3

Documentação do fluxo completo: dataset, Colab, MFCC, quantização INT8,
ESP-IDF/Wokwi, depuração e testes

## 1. Objetivo

O projeto implementa um classificador TinyML capaz de reconhecer os
comandos ABRIR_PORTA e FECHAR_PORTA e usar a classe prevista para
acionar um servo. O objetivo desta documentação é permitir que os demais
membros do grupo compreendam não apenas o resultado final, mas também as
etapas de construção, os problemas encontrados e as decisões técnicas
tomadas.

Figura 1 --- Visão geral do fluxo do projeto.

## 2. Coleta e preparação dos áudios

O dataset inicial contém 60 gravações: 30 da classe ABRIR_PORTA e 30 da
classe FECHAR_PORTA. Os áudios foram padronizados em mono, 16 kHz e
PCM16. O pré-processamento detecta a região de fala, preserva uma margem
de aproximadamente 200 ms e produz uma janela fixa de 2,5 s (40.000
amostras), usando recorte ou padding com zeros quando necessário.

## 3. Extração de MFCC e treinamento no Colab

No Colab, cada áudio é transformado em MFCC com 13 coeficientes e 247
frames, totalizando 3.211 valores. Esses dados alimentam uma CNN pequena
em TensorFlow/Keras. A normalização usa estatísticas calculadas no
conjunto de treinamento.

Os parâmetros de normalização foram preservados em um arquivo NPZ. Os
valores usados no firmware são média = -28,254616 e desvio = 117,86187.
O NPZ é importante porque permite reproduzir no ESP32 a mesma
transformação aplicada durante o treinamento.

## 4. Quantização e exportação do modelo

Após o treinamento, o modelo foi convertido para TensorFlow Lite INT8. A
quantização reduz memória e custo computacional, tornando a inferência
adequada ao microcontrolador. O arquivo .tflite foi então convertido
para model.cc/model.h para ser compilado no firmware. O tensor de
entrada utilizado na aplicação é 1 × 13 × 247 × 1, INT8.

## 5. Migração do pipeline para C/C++ no ESP32

A etapa embarcada não consistiu apenas em carregar o modelo. Foi
necessário reproduzir em C/C++ o mesmo caminho usado no Python:
preparação da janela de áudio, cálculo do MFCC, normalização,
quantização da entrada, execução do TFLite Micro, desquantização da
saída e decisão da classe.

Um problema inicial importante foi o model.cc vindo do exemplo: ele não
representava o modelo treinado no Colab. A solução foi gerar model.cc a
partir do modelo INT8 conhecido, eliminando a inconsistência entre o
ambiente de treinamento e o firmware.

## 6. Problemas técnicos e correções

### Git e build

No ambiente utilizado, o Git 2.42 apresentava operações
recursivas/submódulos que ficavam presas durante configure/build. A
atualização para Git 2.56 resolveu esse bloqueio. Esta observação
descreve o ambiente testado e não deve ser interpretada como uma
incompatibilidade universal do Git 2.42.

### Inferência ANSI C

A execução otimizada do ESP-NN apresentou falha de MaxPool no Wokwi. O
projeto foi configurado para usar a implementação ANSI C do TFLite
Micro.

### FFT ANSI

Durante a validação do MFCC, as chamadas genéricas do ESP-DSP produziram
um espectro divergente no ambiente atual. O uso explícito de
dsps_fft2r_fc32_ansi e dsps_bit_rev_fc32_ansi passou a coincidir com a
referência NumPy.

### Memória

A primeira implementação do MFCC excedeu a memória disponível. O
algoritmo foi reorganizado para evitar matrizes grandes, usar buffers
menores/reutilizáveis e alocar dinamicamente o buffer de áudio
processado.

## 7. Execução no Wokwi

Seis arquivos WAV de teste foram convertidos para arrays PCM16 C++ e
incorporados ao firmware. O menu permite escolher o áudio sem mostrar a
classe antes da inferência, preservando a lógica de teste cego.

Depois da seleção, o firmware executa o pré-processamento, calcula o
MFCC, normaliza e quantiza os 3.211 valores e chama o modelo INT8. A
classe prevista determina a ação do servo.

## 8. Resultados dos seis testes

| Teste \| Classe real \| Classe prevista \| Resultado \|

| --- \| --- \| --- \| --- \|

| 01 \| ABRIR \| ABRIR 99,22% \| Correto \|

| 02 \| ABRIR \| ABRIR 100,00% \| Correto \|

| 03 \| FECHAR \| FECHAR 97,27% \| Correto \|

| 04 \| FECHAR \| FECHAR 97,27% \| Correto \|

| 05 -- ElevenLabs \| FECHAR \| FECHAR 96,88% \| Correto \|

| 06 -- Audacity \| ABRIR \| FECHAR 96,88% \| Incorreto \|

Os quatro primeiros áudios inéditos para a inferência embarcada foram
classificados corretamente. O teste 05, com uma nova voz gerada no
ElevenLabs, também foi correto. O teste 06, gravado por outra pessoa no
Audacity, foi classificado incorretamente como FECHAR_PORTA com 96,88%.
Esse erro é uma evidência útil: o pipeline funciona, mas o conjunto de
treinamento ainda não oferece diversidade suficiente para garantir
generalização entre locutores.

## 9. Interpretação do servo e aplicação física

No simulador, o servo representa o atuador mecânico. Em uma
implementação física, ABRIR_PORTA pode acionar a maçaneta ou o trinco e
retornar à posição inicial; FECHAR_PORTA pode usar uma rotação de 90°
para representar o travamento da fechadura. O movimento exato dependeria
do mecanismo mecânico real.

## 10. Conclusões e próximos passos

O projeto chegou a um pipeline Edge AI completo e executável no
ESP32-S3/Wokwi. A inferência ocorre localmente: o áudio é convertido em
características MFCC, quantizado e classificado no próprio dispositivo,
que então aciona o servo.

-   Ampliar o dataset com vários locutores para as duas classes.

-   Manter os áudios externos atuais fora do treinamento para comparação
    futura.

-   Retreinar e repetir os mesmos testes para medir a melhoria de
    generalização.

-   Depois, reduzir os prints de diagnóstico e preparar a versão final
    para demonstração em vídeo e apresentação.
