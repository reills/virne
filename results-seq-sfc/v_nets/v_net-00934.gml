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
  id 934
  arrival_time 19994.967816120057
  lifetime 896.6193265281689
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 17
    gpu 35
    rom 29
  ]
  node [
    id 1
    label "1"
    cpu 28
    gpu 4
    rom 37
  ]
  node [
    id 2
    label "2"
    cpu 47
    gpu 26
    rom 44
  ]
  edge [
    source 0
    target 1
    bw 25
  ]
  edge [
    source 1
    target 2
    bw 1
  ]
]
