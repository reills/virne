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
  id 1606
  arrival_time 35955.95479000793
  lifetime 2903.729840599536
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 6
    gpu 20
    rom 40
  ]
  node [
    id 1
    label "1"
    cpu 36
    gpu 1
    rom 28
  ]
  node [
    id 2
    label "2"
    cpu 48
    gpu 42
    rom 47
  ]
  node [
    id 3
    label "3"
    cpu 8
    gpu 35
    rom 19
  ]
  node [
    id 4
    label "4"
    cpu 18
    gpu 36
    rom 48
  ]
  node [
    id 5
    label "5"
    cpu 14
    gpu 21
    rom 5
  ]
  node [
    id 6
    label "6"
    cpu 9
    gpu 0
    rom 34
  ]
  node [
    id 7
    label "7"
    cpu 11
    gpu 34
    rom 44
  ]
  node [
    id 8
    label "8"
    cpu 16
    gpu 16
    rom 22
  ]
  node [
    id 9
    label "9"
    cpu 30
    gpu 4
    rom 9
  ]
  node [
    id 10
    label "10"
    cpu 9
    gpu 23
    rom 8
  ]
  node [
    id 11
    label "11"
    cpu 43
    gpu 11
    rom 8
  ]
  node [
    id 12
    label "12"
    cpu 43
    gpu 35
    rom 46
  ]
  node [
    id 13
    label "13"
    cpu 47
    gpu 14
    rom 49
  ]
  node [
    id 14
    label "14"
    cpu 25
    gpu 16
    rom 3
  ]
  edge [
    source 0
    target 1
    bw 30
  ]
  edge [
    source 1
    target 2
    bw 33
  ]
  edge [
    source 2
    target 3
    bw 34
  ]
  edge [
    source 3
    target 4
    bw 4
  ]
  edge [
    source 4
    target 5
    bw 2
  ]
  edge [
    source 5
    target 6
    bw 34
  ]
  edge [
    source 6
    target 7
    bw 23
  ]
  edge [
    source 7
    target 8
    bw 35
  ]
  edge [
    source 8
    target 9
    bw 43
  ]
  edge [
    source 9
    target 10
    bw 12
  ]
  edge [
    source 10
    target 11
    bw 48
  ]
  edge [
    source 11
    target 12
    bw 19
  ]
  edge [
    source 12
    target 13
    bw 4
  ]
  edge [
    source 13
    target 14
    bw 33
  ]
]
