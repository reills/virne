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
  id 360
  arrival_time 6847.352722611999
  lifetime 869.1659678063568
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 20
    gpu 18
    rom 17
  ]
  node [
    id 1
    label "1"
    cpu 47
    gpu 11
    rom 16
  ]
  edge [
    source 0
    target 1
    bw 29
  ]
]
