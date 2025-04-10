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
  id 1001
  arrival_time 21434.156434727603
  lifetime 572.2929688734627
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 10
    gpu 16
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 19
    gpu 10
    rom 45
  ]
  node [
    id 2
    label "2"
    cpu 3
    gpu 44
    rom 7
  ]
  node [
    id 3
    label "3"
    cpu 6
    gpu 11
    rom 46
  ]
  node [
    id 4
    label "4"
    cpu 18
    gpu 15
    rom 10
  ]
  edge [
    source 0
    target 1
    bw 42
  ]
  edge [
    source 1
    target 2
    bw 30
  ]
  edge [
    source 2
    target 3
    bw 31
  ]
  edge [
    source 3
    target 4
    bw 49
  ]
]
