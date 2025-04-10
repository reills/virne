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
  id 483
  arrival_time 8931.287999561995
  lifetime 88.5316785236838
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 32
    gpu 10
    rom 26
  ]
  node [
    id 1
    label "1"
    cpu 4
    gpu 42
    rom 17
  ]
  edge [
    source 0
    target 1
    bw 27
  ]
]
