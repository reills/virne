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
  id 1974
  arrival_time 43135.90952138373
  lifetime 438.5795353965653
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 20
    gpu 31
    rom 12
  ]
  node [
    id 1
    label "1"
    cpu 1
    gpu 44
    rom 8
  ]
  edge [
    source 0
    target 1
    bw 27
  ]
]
