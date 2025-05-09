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
  id 1102
  arrival_time 22963.924460214203
  lifetime 363.97945536019944
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 50
    gpu 11
    rom 44
  ]
  node [
    id 1
    label "1"
    cpu 22
    gpu 49
    rom 19
  ]
  node [
    id 2
    label "2"
    cpu 41
    gpu 41
    rom 32
  ]
  node [
    id 3
    label "3"
    cpu 15
    gpu 33
    rom 26
  ]
  node [
    id 4
    label "4"
    cpu 35
    gpu 47
    rom 42
  ]
  edge [
    source 0
    target 1
    bw 30
  ]
  edge [
    source 1
    target 2
    bw 8
  ]
  edge [
    source 2
    target 3
    bw 43
  ]
  edge [
    source 3
    target 4
    bw 1
  ]
]
