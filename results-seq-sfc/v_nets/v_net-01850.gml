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
  id 1850
  arrival_time 40834.06627963006
  lifetime 2744.208661311786
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 25
    gpu 42
    rom 38
  ]
  node [
    id 1
    label "1"
    cpu 2
    gpu 24
    rom 39
  ]
  node [
    id 2
    label "2"
    cpu 49
    gpu 6
    rom 36
  ]
  node [
    id 3
    label "3"
    cpu 6
    gpu 36
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 4
    gpu 4
    rom 45
  ]
  node [
    id 5
    label "5"
    cpu 21
    gpu 42
    rom 15
  ]
  node [
    id 6
    label "6"
    cpu 7
    gpu 45
    rom 31
  ]
  node [
    id 7
    label "7"
    cpu 18
    gpu 6
    rom 14
  ]
  node [
    id 8
    label "8"
    cpu 49
    gpu 47
    rom 39
  ]
  node [
    id 9
    label "9"
    cpu 26
    gpu 14
    rom 31
  ]
  node [
    id 10
    label "10"
    cpu 10
    gpu 20
    rom 26
  ]
  node [
    id 11
    label "11"
    cpu 26
    gpu 11
    rom 0
  ]
  node [
    id 12
    label "12"
    cpu 23
    gpu 18
    rom 48
  ]
  node [
    id 13
    label "13"
    cpu 3
    gpu 10
    rom 1
  ]
  node [
    id 14
    label "14"
    cpu 10
    gpu 40
    rom 25
  ]
  edge [
    source 0
    target 1
    bw 11
  ]
  edge [
    source 1
    target 2
    bw 9
  ]
  edge [
    source 2
    target 3
    bw 22
  ]
  edge [
    source 3
    target 4
    bw 6
  ]
  edge [
    source 4
    target 5
    bw 30
  ]
  edge [
    source 5
    target 6
    bw 14
  ]
  edge [
    source 6
    target 7
    bw 27
  ]
  edge [
    source 7
    target 8
    bw 41
  ]
  edge [
    source 8
    target 9
    bw 41
  ]
  edge [
    source 9
    target 10
    bw 34
  ]
  edge [
    source 10
    target 11
    bw 14
  ]
  edge [
    source 11
    target 12
    bw 5
  ]
  edge [
    source 12
    target 13
    bw 40
  ]
  edge [
    source 13
    target 14
    bw 12
  ]
]
