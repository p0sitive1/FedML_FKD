# Install FedML
```
pip install fedml
```

# Building server package

```
sh build_mlops_pkg.sh
```

# General file structure

### torch_server.py
main server file, init server 

### FedGen
stores models

### model
stores server-side manager and aggregator

to alter high level server init options, change serverFKD.py

to alter server-side behaviors, change fkd_server_manager.py and fkd_server_aggregator.py