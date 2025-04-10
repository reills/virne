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
  id 545
  arrival_time 10310.886476043017
  lifetime 7.9561471805538275
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 47
    gpu 29
    rom 47
  ]
  node [
    id 1
    label "1"
    cpu 46
    gpu 49
    rom 24
  ]
  node [
    id 2
    label "2"
    cpu 8
    gpu 22
    rom 45
  ]
  node [
    id 3
    label "3"
    cpu 6
    gpu 43
    rom 23
  ]
  edge [
    source 0
    target 1
    bw 43
  ]
  edge [
    source 1
    target 2
    bw 12
  ]
  edge [
    source 2
    target 3
    bw 46
  ]
]
