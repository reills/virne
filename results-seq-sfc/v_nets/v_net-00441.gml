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
  id 441
  arrival_time 8473.693296531808
  lifetime 105.58336132012663
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 50
    gpu 17
    rom 35
  ]
  node [
    id 1
    label "1"
    cpu 38
    gpu 50
    rom 22
  ]
  node [
    id 2
    label "2"
    cpu 17
    gpu 0
    rom 27
  ]
  node [
    id 3
    label "3"
    cpu 41
    gpu 8
    rom 7
  ]
  node [
    id 4
    label "4"
    cpu 19
    gpu 46
    rom 16
  ]
  node [
    id 5
    label "5"
    cpu 38
    gpu 35
    rom 47
  ]
  node [
    id 6
    label "6"
    cpu 6
    gpu 18
    rom 40
  ]
  node [
    id 7
    label "7"
    cpu 9
    gpu 6
    rom 31
  ]
  node [
    id 8
    label "8"
    cpu 19
    gpu 9
    rom 45
  ]
  node [
    id 9
    label "9"
    cpu 38
    gpu 15
    rom 30
  ]
  node [
    id 10
    label "10"
    cpu 14
    gpu 28
    rom 47
  ]
  node [
    id 11
    label "11"
    cpu 14
    gpu 17
    rom 37
  ]
  node [
    id 12
    label "12"
    cpu 42
    gpu 38
    rom 50
  ]
  node [
    id 13
    label "13"
    cpu 40
    gpu 24
    rom 32
  ]
  edge [
    source 0
    target 1
    bw 6
  ]
  edge [
    source 1
    target 2
    bw 16
  ]
  edge [
    source 2
    target 3
    bw 21
  ]
  edge [
    source 3
    target 4
    bw 40
  ]
  edge [
    source 4
    target 5
    bw 1
  ]
  edge [
    source 5
    target 6
    bw 45
  ]
  edge [
    source 6
    target 7
    bw 46
  ]
  edge [
    source 7
    target 8
    bw 10
  ]
  edge [
    source 8
    target 9
    bw 44
  ]
  edge [
    source 9
    target 10
    bw 23
  ]
  edge [
    source 10
    target 11
    bw 11
  ]
  edge [
    source 11
    target 12
    bw 46
  ]
  edge [
    source 12
    target 13
    bw 35
  ]
]
