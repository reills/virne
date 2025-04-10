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
  id 1699
  arrival_time 37694.17757023575
  lifetime 1056.5258067964742
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 9
    gpu 40
    rom 43
  ]
  node [
    id 1
    label "1"
    cpu 16
    gpu 39
    rom 24
  ]
  node [
    id 2
    label "2"
    cpu 12
    gpu 17
    rom 4
  ]
  node [
    id 3
    label "3"
    cpu 27
    gpu 12
    rom 20
  ]
  node [
    id 4
    label "4"
    cpu 13
    gpu 16
    rom 35
  ]
  edge [
    source 0
    target 1
    bw 24
  ]
  edge [
    source 1
    target 2
    bw 46
  ]
  edge [
    source 2
    target 3
    bw 41
  ]
  edge [
    source 3
    target 4
    bw 38
  ]
]
