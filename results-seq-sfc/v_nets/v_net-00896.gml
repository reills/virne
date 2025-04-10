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
  id 896
  arrival_time 19266.862655059544
  lifetime 881.7374929516027
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 48
    gpu 40
    rom 15
  ]
  node [
    id 1
    label "1"
    cpu 44
    gpu 6
    rom 44
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 26
    rom 27
  ]
  node [
    id 3
    label "3"
    cpu 41
    gpu 11
    rom 16
  ]
  node [
    id 4
    label "4"
    cpu 50
    gpu 32
    rom 25
  ]
  node [
    id 5
    label "5"
    cpu 25
    gpu 13
    rom 28
  ]
  node [
    id 6
    label "6"
    cpu 41
    gpu 16
    rom 17
  ]
  node [
    id 7
    label "7"
    cpu 41
    gpu 26
    rom 29
  ]
  node [
    id 8
    label "8"
    cpu 41
    gpu 50
    rom 47
  ]
  node [
    id 9
    label "9"
    cpu 16
    gpu 47
    rom 37
  ]
  node [
    id 10
    label "10"
    cpu 42
    gpu 34
    rom 24
  ]
  node [
    id 11
    label "11"
    cpu 30
    gpu 5
    rom 2
  ]
  node [
    id 12
    label "12"
    cpu 7
    gpu 6
    rom 25
  ]
  node [
    id 13
    label "13"
    cpu 50
    gpu 20
    rom 40
  ]
  edge [
    source 0
    target 1
    bw 38
  ]
  edge [
    source 1
    target 2
    bw 21
  ]
  edge [
    source 2
    target 3
    bw 17
  ]
  edge [
    source 3
    target 4
    bw 23
  ]
  edge [
    source 4
    target 5
    bw 0
  ]
  edge [
    source 5
    target 6
    bw 50
  ]
  edge [
    source 6
    target 7
    bw 17
  ]
  edge [
    source 7
    target 8
    bw 35
  ]
  edge [
    source 8
    target 9
    bw 35
  ]
  edge [
    source 9
    target 10
    bw 46
  ]
  edge [
    source 10
    target 11
    bw 2
  ]
  edge [
    source 11
    target 12
    bw 26
  ]
  edge [
    source 12
    target 13
    bw 4
  ]
]
