# phd-navar

## Configuração do ambiente

Instalar os requirements.txt em um ambiente virtual com **_python3.8.*_**.

```
python -m venv .env
. ./.bin/activate
pip install -r requirements.txt
```
## Treinamento e Teste com base de dados de casas

Baixar a database nesse [link](https://1drv.ms/u/s!AqUPqx8G81xZiawi20d2PucCFrAKzA?e=2MHhua) e extrair para a pasta data.
### Treinamento
```
python keras_efficient.py
```
### Teste
```
python inference.py
```
