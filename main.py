# Run as Administrator only!
# That's because cmd-commands (running below) require Administrator privileges

# ToDo
# Find a way to elevate up to Administrator privileges programmatically

# from open_vpn import OpenVPN
# from elastic_search import ElasticSearch
from elasticsearch.serializer import JSONSerializer


def main():
    # vpn = OpenVPN(way='gui')
    # if vpn.connect():
    #     print('Соединение с OpenVPN успешно установлено. IP-адрес:', vpn.ip_address)
    # else:
    #     print('Соединение с OpenVPN не удалось установить.')
    #     return

    # es = ElasticSearch()
    # if es.connect():
    #     print('Соединение с ElasticSearch успешно установлено.')
    # else:
    #     print('Соединение с ElasticSearch не удалось установить.')
#
    # # Что-то...
    # es.search()

    pass

#
    # if es.disconnect():
    #     print('Соединение с ElasticSearch успешно закрыто.')
    # else:
    #     print('Соединение с ElasticSearch не удалось закрыть.')

    # if vpn.disconnect():
    #     print('Соединение с OpenVPN успешно закрыто.')
    # else:
    #     print('Соединение с OpenVPN не удалось закрыть.')
    #     return




if __name__ == '__main__':
    main()
