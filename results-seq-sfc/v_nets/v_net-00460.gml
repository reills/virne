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
  id 460
  arrival_time 8694.805593153878
  lifetime 476.2642253171972
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 32
    gpu 36
    rom 38
  ]
  node [
    id 1
    label "1"
    cpu 26
    gpu 34
    rom 15
  ]
  node [
    id 2
    label "2"
    cpu 27
    gpu 28
    rom 17
  ]
  node [
    id 3
    label "3"
    cpu 12
    gpu 46
    rom 4
  ]
  node [
    id 4
    label "4"
    cpu 50
    gpu 26
    rom 20
  ]
  node [
    id 5
    label "5"
    cpu 12
    gpu 24
    rom 10
  ]
  node [
    id 6
    label "6"
    cpu 15
    gpu 1
    rom 10
  ]
  node [
    id 7
    label "7"
    cpu 22
    gpu 28
    rom 34
  ]
  node [
    id 8
    label "8"
    cpu 5
    gpu 15
    rom 8
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 20
    rom 24
  ]
  edge [
    source 0
    target 1
    bw 31
  ]
  edge [
    source 1
    target 2
    bw 16
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 25
  ]
  edge [
    source 4
    target 5
    bw 5
  ]
  edge [
    source 5
    target 6
    bw 19
  ]
  edge [
    source 6
    target 7
    bw 20
  ]
  edge [
    source 7
    target 8
    bw 30
  ]
  edge [
    source 8
    target 9
    bw 0
  ]
]
