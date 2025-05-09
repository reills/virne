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
  id 1037
  arrival_time 21924.174339586905
  lifetime 935.5707509109786
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 40
    gpu 0
    rom 47
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 6
    rom 38
  ]
  node [
    id 2
    label "2"
    cpu 24
    gpu 13
    rom 39
  ]
  node [
    id 3
    label "3"
    cpu 27
    gpu 49
    rom 2
  ]
  node [
    id 4
    label "4"
    cpu 11
    gpu 11
    rom 28
  ]
  node [
    id 5
    label "5"
    cpu 42
    gpu 48
    rom 21
  ]
  node [
    id 6
    label "6"
    cpu 18
    gpu 4
    rom 31
  ]
  node [
    id 7
    label "7"
    cpu 33
    gpu 7
    rom 17
  ]
  node [
    id 8
    label "8"
    cpu 23
    gpu 12
    rom 22
  ]
  node [
    id 9
    label "9"
    cpu 16
    gpu 30
    rom 38
  ]
  node [
    id 10
    label "10"
    cpu 25
    gpu 38
    rom 6
  ]
  node [
    id 11
    label "11"
    cpu 26
    gpu 29
    rom 43
  ]
  node [
    id 12
    label "12"
    cpu 6
    gpu 50
    rom 8
  ]
  edge [
    source 0
    target 1
    bw 25
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 48
  ]
  edge [
    source 3
    target 4
    bw 18
  ]
  edge [
    source 4
    target 5
    bw 39
  ]
  edge [
    source 5
    target 6
    bw 42
  ]
  edge [
    source 6
    target 7
    bw 25
  ]
  edge [
    source 7
    target 8
    bw 36
  ]
  edge [
    source 8
    target 9
    bw 19
  ]
  edge [
    source 9
    target 10
    bw 34
  ]
  edge [
    source 10
    target 11
    bw 4
  ]
  edge [
    source 11
    target 12
    bw 18
  ]
]
