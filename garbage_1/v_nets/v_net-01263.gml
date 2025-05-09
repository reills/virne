graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1263
  arrival_time 26569.024353259258
  lifetime 1824.0107570437954
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 15
    gpu 46
    rom 12
  ]
  node [
    id 1
    label "1"
    cpu 14
    gpu 2
    rom 46
  ]
  edge [
    source 0
    target 1
    bw 10
  ]
]
