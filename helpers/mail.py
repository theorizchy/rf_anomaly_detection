import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time

# Email you want to send the update from (only works with gmail)
fromEmail = 'smartcctvp@gmail.com'
# An app password here to avoid storing password in plain text
fromEmailPassword = 'xhncobtyuigkckcy'

# Email you want to send the update to
toEmail = 'cleven.theorizchy@gmail.com'

def sendEmail():
	msgRoot = MIMEMultipart('related')
	msgRoot['Subject'] = '[NOTIFICATION] Security Alert!'
	msgRoot['From'] = fromEmail
	msgRoot['To'] = toEmail
	msgRoot.preamble = 'Hidden Camera may be activated'

    # Create the email body text with the time in bold
	current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	body_text = f"RF Anomaly detector find a potential breach at <b>{current_time}</b>.\n\
                  Hidden camera might be activated on your premises.\n\n\
                  Please disregard this message if you acknowledge such activities.\n\n\
                  Sincerly,\n\
                  Smart RF Detector"
	msgText = MIMEText(body_text, 'html')
	msgRoot.attach(msgText)

	smtp = smtplib.SMTP('smtp.gmail.com', 587)
	smtp.starttls()
	smtp.login(fromEmail, fromEmailPassword)
	smtp.sendmail(fromEmail, toEmail, msgRoot.as_string())
	smtp.quit()
