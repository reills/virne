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
  id 1490
  arrival_time 32946.826418099
  lifetime 162.6294983146646
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 18
    gpu 17
    rom 12
  ]
  node [
    id 1
    label "1"
    cpu 40
    gpu 28
    rom 47
  ]
  node [
    id 2
    label "2"
    cpu 8
    gpu 20
    rom 33
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 12
    rom 15
  ]
  node [
    id 4
    label "4"
    cpu 9
    gpu 17
    rom 1
  ]
  node [
    id 5
    label "5"
    cpu 15
    gpu 5
    rom 11
  ]
  edge [
    source 0
    target 1
    bw 6
  ]
  edge [
    source 1
    target 2
    bw 12
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 43
  ]
  edge [
    source 4
    target 5
    bw 38
  ]
]
