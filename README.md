# FaceMetrics-Project
-----
## Descrição
O objetivo do projeto é coletar métricas do usuário, determinando sua identidade, se está prestando atenção na tela do computador, além de suas emoções.
Existem ainda alarmes caso a pessoa feche os olhos por muito tempo.

Os dados coletados são exportados para um arquivo em formato json.

O código utiliza um módulo de reconhecimento facial da OpenCv, analisa as emoções com auxílio da biblioteca FER e implementa um Face Mesh 3d da biblioteca mediapipe. Por meio deste Face Mesh, consegue detectar a posição da face do usuário e se os seus olhos estão fechados.


## Instruções de uso
### Dependências necessárias
pip install FER

pip install tensorflow

pip install keyboard

### Como utilzar
1. Criar uma pasta chamada "users" - é nela que as fotos de cadastro de cada usuário ficarão guardadas
2. Rodar o arquivo cadastro.py --> 20 fotos serão tiradas para o reconhecimento facial no vídeo
3. Rodar o arquivo face_recognition_training.py para treinar o modelo de reconhecimento facial
4. Rodar o código principal no arquivo [face_metrics_final.py](../master/face_metrics_final.py)
5. Clicar na tecla H para visualizar as métricas em tempo real no terminal ou S para salvar em um arquivo
6. Depois, as métricas coletadas podem ser consultadas no arquivo history.txt

### Relatório e explicações 
[viscomp-p2.pdf](../master/viscomp-p2.pdf)
