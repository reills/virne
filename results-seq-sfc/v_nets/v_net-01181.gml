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
  id 1181
  arrival_time 24392.6549866605
  lifetime 4991.85275537442
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 31
    gpu 13
    rom 22
  ]
  node [
    id 1
    label "1"
    cpu 7
    gpu 18
    rom 32
  ]
  node [
    id 2
    label "2"
    cpu 36
    gpu 12
    rom 22
  ]
  node [
    id 3
    label "3"
    cpu 13
    gpu 46
    rom 50
  ]
  node [
    id 4
    label "4"
    cpu 46
    gpu 10
    rom 47
  ]
  node [
    id 5
    label "5"
    cpu 18
    gpu 47
    rom 17
  ]
  node [
    id 6
    label "6"
    cpu 43
    gpu 5
    rom 41
  ]
  node [
    id 7
    label "7"
    cpu 5
    gpu 1
    rom 20
  ]
  node [
    id 8
    label "8"
    cpu 25
    gpu 10
    rom 40
  ]
  node [
    id 9
    label "9"
    cpu 5
    gpu 40
    rom 46
  ]
  node [
    id 10
    label "10"
    cpu 28
    gpu 5
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 40
  ]
  edge [
    source 1
    target 2
    bw 4
  ]
  edge [
    source 2
    target 3
    bw 8
  ]
  edge [
    source 3
    target 4
    bw 45
  ]
  edge [
    source 4
    target 5
    bw 2
  ]
  edge [
    source 5
    target 6
    bw 3
  ]
  edge [
    source 6
    target 7
    bw 19
  ]
  edge [
    source 7
    target 8
    bw 40
  ]
  edge [
    source 8
    target 9
    bw 2
  ]
  edge [
    source 9
    target 10
    bw 7
  ]
]
