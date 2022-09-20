import os.path
import subprocess
import time
import signal
from telnetlib import Telnet


class OpenVPN:
    def __init__(self, way='manual'):
        self.way = way
        self.openvpn_path = 'C:/Program Files/OpenVPN'
        self.config_name = 'config'
        self.log = f'{self.openvpn_path}/log/{self.config_name}.log'
        self.cmd_shell = None
        self.exit_event_name = 'CTRL_BREAK_EVENT'
        self.management_ip_address = '127.0.0.1'  # localhost
        self.management_port = 25340
        self.management_interface = None
        self.ip_address = None

    def clear(self):
        self.way = 'manual'
        self.cmd_shell = None
        self.management_interface = None
        self.ip_address = None

    def connect(self):
        if self.is_connected():
            self._set_ip_address()
            self._ping()
            return True
        if self.way == 'manual':
            cmd_connect = f'"{self.openvpn_path}/bin/openvpn.exe"' \
                          f' --log "{self.log}"' \
                          f' --config "{self.openvpn_path}/config/{self.config_name}.ovpn"' \
                          f' --service {self.exit_event_name} 0' \
                          f' --auth-retry interact' \
                          f' --management {self.management_ip_address} {self.management_port}' \
                          f' --management-query-passwords' \
                          f' --management-hold'
        elif self.way == 'easy':
            cmd_connect = f'"{self.openvpn_path}/bin/openvpn.exe"' \
                          f' --log "{self.log}"' \
                          f' --config "{self.openvpn_path}/config/{self.config_name}.ovpn"'
        elif self.way == 'gui':
            cmd_connect = f'"{self.openvpn_path}/bin/openvpn-gui.exe" --command connect "{self.config_name}"' \
                          f' --log_dir "{self.openvpn_path}/log"'
        else:
            return False

        print('Запуск команды подключения...')
        self.cmd_shell = subprocess.Popen(cmd_connect, shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        time.sleep(1)
        if self.way == 'manual':
            print('Запуск Management Interface...')
            self.management_interface = Telnet(self.management_ip_address, self.management_port)
            self.management_interface.read_until(b'\n')
            if self.management_interface.read_until(b'\n').decode('utf-8').__contains__('HOLD'):
                self.management_interface.write(b'state on\n')
                # self.management_interface.read_until(b'state on\n')
                if self.management_interface.read_until(b'\n').decode('utf-8').__contains__('SUCCESS'):
                    self.management_interface.write(b'hold off\n')
                    # self.management_interface.read_until(b'hold off\n')
                    if self.management_interface.read_until(b'\n').decode('utf-8').__contains__('SUCCESS'):
                        self.management_interface.write(b'hold release\n')
                        # self.management_interface.read_until(b'hold release\n')
                        if self.management_interface.read_until(b'\n').decode('utf-8').__contains__('SUCCESS'):
                            pass
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        # Ожидание подключения
        for i in range(3):
            time.sleep(10)
            if self.is_connected():
                self._set_ip_address()
                self._ping()
                return True
        return False

    def disconnect(self):
        if not self.is_connected():
            return True
        print('Запуск команды отсоединения...')
        if self.way == 'manual':
            print('Посыл сигнала завершения в cmd...')
            self.cmd_shell.send_signal(signal.CTRL_BREAK_EVENT)
            print('Ожидание завершения процесса cmd...')
            self.cmd_shell.wait()
            print('Ожидание завершения процесса Management Interface')
            self.management_interface.close()
        elif self.way == 'easy':
            print('Посыл сигнала завершения в cmd...')
            self.cmd_shell.send_signal(signal.CTRL_BREAK_EVENT)
            print('Ожидание завершения процесса cmd')
            self.cmd_shell.wait()
        elif self.way == 'gui':
            cmd_disconnect = f'"{self.openvpn_path}/bin/openvpn-gui.exe" --command disconnect "{self.config_name}"'
            print('Посыл команды завершения...')
            subprocess.run(cmd_disconnect)
        else:
            return False
        self.clear()
        return not self.is_connected()

    def is_connected(self):
        if not os.path.exists(self.log):
            print('Лог отсутствует.')
            return False
        connected = False
        with open(self.log, 'r') as file:
            while True:
                s = file.readline()
                if not s:
                    break
                if s.__contains__('MANAGEMENT'):
                    if s.__contains__('CONNECTED') and s.__contains__('SUCCESS'):
                        connected = True
                    elif s.__contains__('EXITING') and s.__contains__('SIGTERM'):
                        connected = False
        return connected

    def _set_ip_address(self):
        if not os.path.exists(self.log) or not self.is_connected():
            print('Лог отсутствует.')
            return
        with open(self.log, 'r') as file:
            while True:
                s = file.readline()
                if not s:
                    break
                if s.__contains__('MANAGEMENT') and s.__contains__('CONNECTED') and s.__contains__('SUCCESS'):
                    self.ip_address = s.split(',')[3]

    def _ping(self):
        proc = subprocess.run(f'ping {self.ip_address}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc.stdout.decode('utf-8')
