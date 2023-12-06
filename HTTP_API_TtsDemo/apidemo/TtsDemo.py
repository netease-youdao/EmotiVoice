import requests

from utils.AuthV3Util import addAuthParams

# 您的应用ID
APP_KEY = ''
# 您的应用密钥
APP_SECRET = ''

# 合成音频保存路径, 例windows路径：PATH = "C:\\tts\\media.mp3"
PATH = 'EmotiVoice-8051.mp3'


def createRequest():
    '''
    note: 将下列变量替换为需要请求的参数
    '''
    q = 'Emoti-Voice - a Multi-Voice and Prompt-Controlled T-T-S Engine，大家好'
    voiceName = 'Maria Kasper' # 'Cori Samuel'
    format = 'mp3'

    data = {'q': q, 'voiceName': voiceName, 'format': format}

    addAuthParams(APP_KEY, APP_SECRET, data)

    header = {'Content-Type': 'application/x-www-form-urlencoded'}
    res = doCall('https://openapi.youdao.com/ttsapi', header, data, 'post')
    saveFile(res)


def doCall(url, header, params, method):
    if 'get' == method:
        return requests.get(url, params)
    elif 'post' == method:
        return requests.post(url, params, header)


def saveFile(res):
    contentType = res.headers['Content-Type']
    if 'audio' in contentType:
        fo = open(PATH, 'wb')
        fo.write(res.content)
        fo.close()
        print('save file path: ' + PATH)
    else:
        print(str(res.content, 'utf-8'))

# 网易有道智云语音合成服务api调用demo
# api接口: https://openapi.youdao.com/ttsapi
if __name__ == '__main__':
    createRequest()
