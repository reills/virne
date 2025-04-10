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
  id 1816
  arrival_time 40262.34751395729
  lifetime 1493.5202852006107
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 31
    gpu 47
    rom 48
  ]
  node [
    id 1
    label "1"
    cpu 32
    gpu 25
    rom 17
  ]
  node [
    id 2
    label "2"
    cpu 39
    gpu 21
    rom 3
  ]
  node [
    id 3
    label "3"
    cpu 44
    gpu 0
    rom 47
  ]
  node [
    id 4
    label "4"
    cpu 23
    gpu 49
    rom 34
  ]
  node [
    id 5
    label "5"
    cpu 10
    gpu 44
    rom 23
  ]
  node [
    id 6
    label "6"
    cpu 40
    gpu 42
    rom 44
  ]
  node [
    id 7
    label "7"
    cpu 8
    gpu 42
    rom 37
  ]
  node [
    id 8
    label "8"
    cpu 11
    gpu 18
    rom 9
  ]
  edge [
    source 0
    target 1
    bw 4
  ]
  edge [
    source 1
    target 2
    bw 38
  ]
  edge [
    source 2
    target 3
    bw 16
  ]
  edge [
    source 3
    target 4
    bw 12
  ]
  edge [
    source 4
    target 5
    bw 26
  ]
  edge [
    source 5
    target 6
    bw 29
  ]
  edge [
    source 6
    target 7
    bw 8
  ]
  edge [
    source 7
    target 8
    bw 1
  ]
]
