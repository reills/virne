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
  id 1860
  arrival_time 40931.70484449962
  lifetime 65.54290448354212
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 4
    gpu 22
    rom 42
  ]
  node [
    id 1
    label "1"
    cpu 3
    gpu 43
    rom 1
  ]
  node [
    id 2
    label "2"
    cpu 24
    gpu 14
    rom 33
  ]
  node [
    id 3
    label "3"
    cpu 34
    gpu 2
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 7
  ]
  edge [
    source 1
    target 2
    bw 46
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
]
