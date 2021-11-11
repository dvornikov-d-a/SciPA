# Run as Administrator only!
# That's because cmd-commands (running below) require Administrator privileges

from open_vpn import OpenVPN
from elastic_search import ElasticSearch


def elevate():
    # ToDo
    # Find a way to elevate up to Administrator privileges programmatically
    pass


def download():
    elevate()

    vpn = OpenVPN(way='gui')
    if vpn.connect():
        print('Соединение с OpenVPN успешно установлено. IP-адрес:', vpn.ip_address)
    else:
        print('Соединение с OpenVPN не удалось установить.')
        return

    es = ElasticSearch()
    if es.connect():
        print('Соединение с ElasticSearch успешно установлено.')
    else:
        print('Соединение с ElasticSearch не удалось установить.')
        return

    es.search_and_save()

    if es.disconnect():
        print('Соединение с ElasticSearch успешно закрыто.')
    else:
        print('Соединение с ElasticSearch не удалось закрыть.')

    if vpn.disconnect():
        print('Соединение с OpenVPN успешно закрыто.')
    else:
        print('Соединение с OpenVPN не удалось закрыть.')
