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
  id 1694
  arrival_time 37649.425627153956
  lifetime 349.0751891777749
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 26
    gpu 13
    rom 46
  ]
  node [
    id 1
    label "1"
    cpu 5
    gpu 4
    rom 43
  ]
  node [
    id 2
    label "2"
    cpu 19
    gpu 31
    rom 16
  ]
  node [
    id 3
    label "3"
    cpu 5
    gpu 18
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 44
    gpu 39
    rom 47
  ]
  node [
    id 5
    label "5"
    cpu 49
    gpu 16
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 39
  ]
  edge [
    source 1
    target 2
    bw 25
  ]
  edge [
    source 2
    target 3
    bw 21
  ]
  edge [
    source 3
    target 4
    bw 38
  ]
  edge [
    source 4
    target 5
    bw 12
  ]
]
