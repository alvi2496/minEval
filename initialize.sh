./config/environment/setup.sh

source ~/pyvirenvs/mineval/bin/activate

./config/dependencies/install.sh

PROJECT_PATH=`pwd`

cat <<EOF >$PROJECT_PATH/activate.sh
source ~/pyvirenvs/mineval/bin/activate
./config/dependencies/install.sh
EOF

chmod 777 activate.sh