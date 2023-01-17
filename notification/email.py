import smtplib, ssl
from email.message import EmailMessage
from email_args import SOURCE_EMAIL, PASSWORD, DESTINATION_EMAIL, HEADER, SUBJECT

PORT = 465 # For SSL
'''
function to send notification email from SOURCE_EMAIL to DESTINATION_EMAIL
args expects a dict of metrics in form of {'metric_name': metric}. 
Default subject and headers can be defined in email_args.py
'''
def send_email(args: dict=None, subject: str = SUBJECT, header:str = HEADER):

    context = ssl.create_default_context()
    msg = EmailMessage()
    msg.set_content(f'{header}\n{dict_to_str(args)}')

    msg['Subject'] = subject
    msg['From'] = SOURCE_EMAIL
    msg['To'] = DESTINATION_EMAIL

    with smtplib.SMTP_SSL("smtp.gmail.com", PORT, context=context) as server:
        server.login(SOURCE_EMAIL, PASSWORD)
        server.send_message(msg)

def dict_to_str(d):
    return '' if not d else '\n'.join([f'{k}: {v}' for k,v in d.items()])