import getopt
import os.path as op
import shutil
import smtplib
import ssl
import sys
import time
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json


# Configure Mail Notifyers
port = 465  # For SSL
with open('./config.json', mode='r') as fi:
    config = json.load(fi)
    if 'Notifyer' not in config:
        print("System notifyer unconfigured")
        exit(100)

config = config['Notifyer']
password = config['password']
sender_email = config['sender_email']
receiver_email = config['receiver_email']
file_name = 'default'
context = ssl.create_default_context()


def compose_from_log(new_content, interval, disk_info, file_name):
    message = MIMEMultipart("alternative")
    message["Subject"] = "[Micipsa Pipeline] New log from {} at {}".format(
        file_name, time.ctime())
    message["From"] = "Bot CAMS <{}>".format(sender_email)
    message["To"] = "Master <{}>".format(receiver_email)
    text = """\
{} lines of new log generated since last {} seconds.
Disk space: total {} G, used {} G, free {} G.
Log from: {}

{}""".format(len(new_content), interval, disk_info[0], disk_info[1], disk_info[2], file_name, ''.join(new_content))
    part1 = MIMEText(text, "plain")
    message.attach(part1)
    return message.as_string()


def send_mail_log(subject, content,
                  context=context, sender_email=sender_email, receiver_email=receiver_email,
                  port=port, password=password):
    disk_info = np.array(shutil.disk_usage('/'))/1073741824
    message = MIMEMultipart("alternative")
    message["Subject"] = "[Micipsa Pipeline] New log from {} at {}".format(
        subject, time.ctime())
    message["From"] = "Bot CAMS <{}>".format(sender_email)
    message["To"] = "Master <{}>".format(receiver_email)
    text = """\
{} generated a new log.
Disk space: total {} G, used {} G, free {} G.

{}""".format(subject, disk_info[0], disk_info[1], disk_info[2], ''.join(content))
    part1 = MIMEText(text, "plain")
    message.attach(part1)

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("botnicolasien@gmail.com", password)
        server.sendmail(sender_email, receiver_email, message.as_string())

    return


if __name__ == '__main__':
    # parse command line
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "f:i:",
                                   ["file_to_watch=",
                                    "interval="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ('-f', '--file_to_watch'):
            file_path = a
        elif o in ('-i', '--interval'):
            interval = int(a)

    try:
        f = open(file_path, 'r')
    except FileNotFoundError as err:
        print(err)
        sys.exit(3)

    file_name = op.basename(file_path)
    current_interval = interval
    while True:
        lines = []
        while True:
            line = f.readline()
            if not line:
                break
            else:
                lines.append(line)
        if len(lines) == 0:
            current_interval += interval
            time.sleep(interval)
        else:
            disk_info = np.array(shutil.disk_usage('/'))/1073741824
            with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                server.login("botnicolasien@gmail.com", password)
                server.sendmail(sender_email, receiver_email, compose_from_log(
                    lines, current_interval, disk_info, file_name))
            current_interval = interval
