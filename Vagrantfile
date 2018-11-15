PYTHON_V = File.open("./.python-version") { |f| f.read }.chomp

$apt_script = <<SCRIPT
  sudo apt-get update
  sudo apt-get install -y git-core curl zlib1g-dev build-essential libssl-dev libreadline-dev libyaml-dev libsqlite3-dev \
   sqlite3 libxml2-dev libxslt1-dev libcurl4-openssl-dev python-software-properties libffi-dev libmysqlclient-dev
SCRIPT

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/xenial64"
  # config.vm.box_check_update = false
  # config.vm.network "forwarded_port", guest: 80, host: 8080
  # config.vm.network "forwarded_port", guest: 80, host: 8080, host_ip: "127.0.0.1"
  # config.vm.network "private_network", ip: "192.168.33.10"
  # config.vm.network "public_network"
  config.vm.synced_folder ".", "/home/minEval"
  config.vm.provider "virtualbox" do |vb|
    vb.gui = false
    vb.memory = 1024
  end

  config.vm.provision :shell, inline: $apt_script
end
