apt update
sleep 1
apt install curl mtr python3-dev build-essential libssl-dev libffi-dev -y
sleep 3
pip install nornir nornir-utils
sleep 1
pip install pandas
sleep 1
pip install pynetbox
sleep 1
pip install ripe.atlas.cousteau
sleep 1
pip install ripe.atlas.sagan
sleep 1
pip install nornir-jinja2
sleep 1
pip install nornir-netbox
sleep 1
pip install nornir-scrapli
sleep 1
curl -s https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh > gitlab-runner-install.sh
chmod +x gitlab-runner-install.sh
sleep 2
os=ubuntu dist=jammy ./gitlab-runner-install.sh
sleep 3
apt-get install -y gitlab-runner
sleep 5
apt install python3-psycopg2 -y
sleep 1
apt install postgresql-client -y
sleep 1
apt install python3-numpy -y
sleep 1
