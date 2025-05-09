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
  id 702
  arrival_time 14774.237736855568
  lifetime 91.3195249585914
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 4
    gpu 32
    rom 31
  ]
  node [
    id 1
    label "1"
    cpu 46
    gpu 15
    rom 39
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 22
    rom 29
  ]
  node [
    id 3
    label "3"
    cpu 13
    gpu 48
    rom 8
  ]
  node [
    id 4
    label "4"
    cpu 8
    gpu 2
    rom 14
  ]
  node [
    id 5
    label "5"
    cpu 0
    gpu 17
    rom 41
  ]
  node [
    id 6
    label "6"
    cpu 3
    gpu 21
    rom 45
  ]
  node [
    id 7
    label "7"
    cpu 1
    gpu 36
    rom 1
  ]
  node [
    id 8
    label "8"
    cpu 43
    gpu 28
    rom 5
  ]
  node [
    id 9
    label "9"
    cpu 23
    gpu 29
    rom 34
  ]
  node [
    id 10
    label "10"
    cpu 38
    gpu 50
    rom 37
  ]
  node [
    id 11
    label "11"
    cpu 21
    gpu 44
    rom 19
  ]
  node [
    id 12
    label "12"
    cpu 1
    gpu 17
    rom 29
  ]
  node [
    id 13
    label "13"
    cpu 25
    gpu 21
    rom 43
  ]
  node [
    id 14
    label "14"
    cpu 22
    gpu 38
    rom 20
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 11
  ]
  edge [
    source 2
    target 3
    bw 10
  ]
  edge [
    source 3
    target 4
    bw 4
  ]
  edge [
    source 4
    target 5
    bw 12
  ]
  edge [
    source 5
    target 6
    bw 40
  ]
  edge [
    source 6
    target 7
    bw 38
  ]
  edge [
    source 7
    target 8
    bw 40
  ]
  edge [
    source 8
    target 9
    bw 7
  ]
  edge [
    source 9
    target 10
    bw 0
  ]
  edge [
    source 10
    target 11
    bw 18
  ]
  edge [
    source 11
    target 12
    bw 42
  ]
  edge [
    source 12
    target 13
    bw 47
  ]
  edge [
    source 13
    target 14
    bw 43
  ]
]
