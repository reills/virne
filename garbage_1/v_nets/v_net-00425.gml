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
  id 425
  arrival_time 8329.885325434934
  lifetime 3460.02029173451
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 12
    gpu 32
    rom 6
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 10
    rom 16
  ]
  node [
    id 2
    label "2"
    cpu 22
    gpu 33
    rom 8
  ]
  node [
    id 3
    label "3"
    cpu 9
    gpu 36
    rom 28
  ]
  node [
    id 4
    label "4"
    cpu 13
    gpu 46
    rom 32
  ]
  edge [
    source 0
    target 1
    bw 0
  ]
  edge [
    source 1
    target 2
    bw 46
  ]
  edge [
    source 2
    target 3
    bw 29
  ]
  edge [
    source 3
    target 4
    bw 11
  ]
]
